"""
Parallel Bedrock LLM Evaluator with Real-time Progress Updates

Uses AWS Bedrock with concurrent requests for fast paper analysis.
Provides real-time status updates for web UI.
"""

import os
import json
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading

import boto3
from botocore.exceptions import ClientError

from src.core.markdown_parser import DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer
from src.core.analyzer import CriterionResult, PaperResult


@dataclass
class PaperStatus:
    """Real-time status of paper analysis"""
    doc_id: str
    title: str
    status: str  # "pending", "running", "completed", "failed"
    progress: int  # 0-100
    criteria_results: Dict[str, str]  # criterion_name -> "Yes"/"No"/"Unknown"
    overall_score: float
    recommendation: str
    error: Optional[str] = None


class ParallelBedrockEvaluator:
    """Parallel Bedrock evaluator with real-time updates"""
    
    def __init__(self,
                 document_db: DocumentDatabase,
                 vector_store: SimpleVectorStore,
                 region: str = "us-east-1",
                 model_id: str = "us.amazon.nova-micro-v1:0",
                 max_workers: int = 10):
        self.document_db = document_db
        self.vector_store = vector_store
        self.criteria_analyzer = CriteriaAnalyzer(vector_store)
        
        # AWS Bedrock client
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)
        self.model_id = model_id
        self.max_workers = max_workers
        
        # Progress tracking
        self.paper_statuses: Dict[str, PaperStatus] = {}
        self.status_lock = threading.Lock()
        
        # Criteria definitions
        self.criteria = {
            'pytorch': {
                'name': 'PyTorch Framework',
                'description': 'Uses PyTorch framework',
                'required': True,
                'weight': 1.0,
                'keywords': ['pytorch', 'torch', 'torchvision', 'torch.nn']
            },
            'supervised': {
                'name': 'Supervised Learning',
                'description': 'Focuses on supervised learning',
                'required': True,
                'weight': 1.0,
                'keywords': ['supervised', 'classification', 'labeled', 'ground truth']
            },
            'small_dataset': {
                'name': 'Small Dataset',
                'description': 'Works with ≤100K samples',
                'required': False,
                'weight': 0.6,
                'keywords': ['cifar', 'mnist', 'small dataset', 'few-shot']
            },
            'quick_training': {
                'name': 'Quick Training',
                'description': '≤24 hours on single GPU',
                'required': False,
                'weight': 0.4,
                'keywords': ['single gpu', 'fast training', 'efficient', 'lightweight']
            },
            'has_repo': {
                'name': 'Public Repository',
                'description': 'Has public code repository',
                'required': True,
                'weight': 1.0,
                'keywords': ['github', 'code available', 'repository', 'gitlab']
            }
        }
    
    def _call_bedrock(self, prompt: str) -> str:
        """Call AWS Bedrock with error handling - Amazon Nova format"""
        try:
            # Amazon Nova API format
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 500,
                    "temperature": 0.1,
                    "topP": 1.0
                }
            }
            
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Parse Amazon Nova response format
            if 'output' in response_body and 'message' in response_body['output']:
                content = response_body['output']['message']['content']
                if isinstance(content, list) and len(content) > 0:
                    return content[0]['text']
            
            # Fallback
            return str(response_body)
                
        except ClientError as e:
            raise Exception(f"Bedrock API error: {e}")
    
    def _evaluate_criterion(self, doc_id: str, criterion_name: str, content: str) -> CriterionResult:
        """Evaluate single criterion using Bedrock"""
        criterion_def = self.criteria[criterion_name]
        
        prompt = f"""Analyze this research paper content to determine if it meets the criterion: {criterion_def['description']}

Paper content:
{content[:3000]}

Look for: {', '.join(criterion_def['keywords'])}

Respond ONLY with a JSON object in this format:
{{"answer": "Yes|No|Unknown", "confidence": 0.0-1.0, "evidence": "brief quote or finding"}}"""
        
        try:
            response = self._call_bedrock(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                answer = data.get("answer", "Unknown")
                confidence = float(data.get("confidence", 0.5))
                evidence = data.get("evidence", "")
            else:
                # Fallback parsing
                response_lower = response.lower()
                if "yes" in response_lower:
                    answer = "Yes"
                    confidence = 0.7
                elif "no" in response_lower:
                    answer = "No"
                    confidence = 0.7
                else:
                    answer = "Unknown"
                    confidence = 0.3
                evidence = response[:200]
            
            return CriterionResult(
                name=criterion_name,
                answer=answer,
                confidence=confidence,
                evidence=evidence,
                sections=[]
            )
            
        except Exception as e:
            print(f"Error evaluating {criterion_name} for {doc_id}: {e}")
            return CriterionResult(
                name=criterion_name,
                answer="Unknown",
                confidence=0.0,
                evidence=f"Error: {str(e)}",
                sections=[]
            )
    
    def analyze_paper(self, doc_id: str, update_callback=None) -> Optional[PaperResult]:
        """Analyze single paper with progress updates - ALL CRITERIA AT ONCE"""
        # Get document
        doc_meta = self.document_db.get_document(doc_id)
        if not doc_meta:
            return None
        
        sections = self.document_db.get_document_sections(doc_id)
        if not sections:
            return None
        
        title = doc_meta['title']
        
        # Initialize status
        with self.status_lock:
            self.paper_statuses[doc_id] = PaperStatus(
                doc_id=doc_id,
                title=title,
                status="running",
                progress=0,
                criteria_results={},
                overall_score=0.0,
                recommendation="Pending"
            )
        
        if update_callback:
            update_callback(doc_id, "running", 0)
        
        print(f"🔍 Analyzing: {title}")
        
        # COLLECT RELEVANT CONTENT FOR ALL CRITERIA AT ONCE
        all_content = {}
        for criterion_name in self.criteria.keys():
            relevant_sections = self.criteria_analyzer.get_section_content_for_criteria(
                doc_id, criterion_name, top_k=2
            )
            
            content_parts = []
            for section_info in relevant_sections:
                key = section_info['section_key']
                if key in sections:
                    content_parts.append(f"{sections[key][:500]}")
            
            all_content[criterion_name] = "\n".join(content_parts) if content_parts else str(sections)[:1000]
        
        # BUILD ONE MEGA PROMPT WITH DETAILED REASONING INSTRUCTIONS
        mega_prompt = f"""You are an expert AI researcher. Analyze this paper against 5 criteria using DEEP REASONING.

Paper Title: {title}

================================================================================
DETAILED REASONING FRAMEWORK FOR EACH CRITERION
================================================================================

═══════════════════════════════════════════════════════════════════════════════
CRITERION 1: PYTORCH FRAMEWORK - MULTI-STEP REASONING
═══════════════════════════════════════════════════════════════════════════════

CORE QUESTION: Does this paper use PyTorch, making it REPRODUCIBLE for PyTorch users?

═══ REASONING FRAMEWORK ═══

STEP 1: Look for explicit framework mentions

CLEAR PYTORCH (Answer: YES):
✓ Explicit: "We use PyTorch", "implemented in PyTorch"
✓ Code snippets: "import torch", "nn.Module", "torch.optim"
✓ Repository: GitHub repo with PyTorch code
✓ Requirements: "torch>=1.7", "torchvision"
✓ Comparisons: "Unlike TensorFlow implementations, we use PyTorch"

CLEAR NON-PYTORCH (Answer: NO):
✗ Explicit: "TensorFlow", "JAX", "Keras", "MXNet", "Caffe"
✗ Code using other frameworks exclusively
✗ "Implemented in TensorFlow 2.x"

STEP 2: Infer from context if not explicit

LIKELY PYTORCH INDICATORS:
✓ Recent papers (2020+) with DL implementations (PyTorch is dominant)
✓ Academic research code (PyTorch is standard in academia)
✓ Computer vision/NLP research (PyTorch very common)
✓ GitHub repos from research labs (often PyTorch)
✓ Papers from conferences where PyTorch dominates

LIKELY OTHER FRAMEWORKS:
✗ Explicitly production/industry focused (may use TensorFlow)
✗ Google affiliations + mentions "our framework" (likely JAX/TF)
✗ Very old papers (2017-2018) may use TensorFlow 1.x
✗ Specific framework-dependent features mentioned

STEP 3: Consider practical reproducibility

REPRODUCIBLE IN PYTORCH (Answer: YES even if ambiguous):
✓ Framework-agnostic algorithms that work in any framework
✓ Mathematical descriptions allowing PyTorch implementation
✓ Open source code that could be ported to PyTorch easily
✓ Standard architectures (ResNet, Transformer) available in PyTorch

NOT SPECIFIC TO OTHER FRAMEWORK (Lean toward YES):
✓ If method is general and not tied to specific framework features
✓ If no framework mentioned but likely PyTorch from context
✓ If mathematical/algorithmic focus, framework doesn't matter much

═══ KEY PRINCIPLE ═══

Answer YES if:
- Explicitly uses PyTorch
- Likely uses PyTorch from context (recent, academic, vision/NLP)
- Framework-agnostic approach (works in any framework)
- No explicit use of other frameworks

Answer NO if:
- Explicitly uses non-PyTorch framework (TensorFlow, JAX, etc.)
- Framework-specific features that require non-PyTorch

Answer UNKNOWN if:
- No framework mentioned, can't infer
- Theoretical paper with no implementation

Relevant content:
{all_content['pytorch'][:1200]}

═══════════════════════════════════════════════════════════════════════════════
CRITERION 2: SUPERVISED LEARNING - DETAILED REASONING GUIDE
═══════════════════════════════════════════════════════════════════════════════

CORE QUESTION: Does this method learn from LABELED EXAMPLES to make predictions?

FUNDAMENTAL DEFINITION:
Supervised learning = Learning a function f: X → Y where:
- X = inputs (images, text, graphs, SAT formulas, etc.)
- Y = LABELS (categories, values, solutions, correctness, etc.)
- Training uses (X, Y) pairs where Y is KNOWN GROUND TRUTH

═══ STEP 1: IDENTIFY THE LEARNING PARADIGM ═══

SUPERVISED LEARNING EXAMPLES (Answer: YES):

1. IMAGE CLASSIFICATION
   - Input: images, Label: class (cat/dog/airplane)
   - Datasets: CIFAR-10, ImageNet, MNIST
   - Key: Images are LABELED with ground truth categories

2. OBJECT DETECTION / SEGMENTATION
   - Input: image, Label: bounding boxes/masks with class labels
   - Datasets: COCO, Pascal VOC
   - Key: Annotations are PROVIDED as training labels

3. REGRESSION TASKS
   - Input: features, Label: continuous value
   - Example: House price prediction, age estimation
   - Key: True values are KNOWN during training

4. SEQUENCE PREDICTION WITH LABELS
   - Input: sequence, Label: next token, translation, summary
   - Example: Machine translation (source → target pairs)
   - Key: Target sequences are PROVIDED

5. GRAPH/STRUCTURE PREDICTION WITH LABELS
   - Input: graph/formula, Label: property/solution/classification
   - Examples:
     * SAT solving: formula → satisfiability label (SAT/UNSAT)
     * Graph classification: graph → category label
     * Theorem proving: conjecture → provability label
   - Key: Training on LABELED instances where outcome is KNOWN

6. TIME SERIES WITH KNOWN TARGETS
   - Input: historical data, Label: future value or classification
   - Key: Ground truth future values used for training

7. RANKING/RECOMMENDATION WITH FEEDBACK
   - Input: query/user, Label: relevance scores, clicks, ratings
   - Key: User feedback or expert annotations as labels

═══ STEP 2: DIFFERENTIATE FROM NON-SUPERVISED ═══

NOT SUPERVISED - SELF-SUPERVISED (Answer: NO):
✗ Contrastive learning (SimCLR, MoCo) - creates own pretext tasks
✗ Masked language modeling (BERT) - masks tokens to predict
✗ Rotation prediction, colorization - self-created tasks
✗ Key: Labels are ARTIFICIALLY CREATED, not human-provided ground truth

NOT SUPERVISED - UNSUPERVISED (Answer: NO):
✗ Clustering (K-means, DBSCAN) - no labels at all
✗ Dimensionality reduction (PCA, t-SNE) - no labels
✗ Autoencoders (reconstruction) - no external labels
✗ Anomaly detection without labeled anomalies
✗ Key: NO labels used, only data structure

NOT SUPERVISED - REINFORCEMENT LEARNING (Answer: NO):
✗ Q-learning, policy gradients, actor-critic
✗ Game playing (AlphaGo, Atari games) learning from rewards
✗ Robotics learning from trial and error
✗ Key: Learns from REWARDS/PENALTIES, not labeled examples

NOT SUPERVISED - SEMI-SUPERVISED (Answer: UNKNOWN or NO):
? Uses small labeled set + large unlabeled set
? May use pseudo-labeling or consistency regularization
? Answer NO unless supervised component is DOMINANT

═══ STEP 3: SUPERVISED INDICATORS TO LOOK FOR ═══

STRONG SUPERVISED INDICATORS:
✓ "training set", "validation set", "test set" with LABELS
✓ "annotated data", "labeled examples", "ground truth"
✓ Loss functions: cross-entropy, categorical loss, MSE with targets
✓ Metrics: accuracy, precision, recall, F1, AUC (require labels)
✓ "classification", "regression", "prediction" tasks
✓ Datasets known to be labeled: CIFAR, MNIST, ImageNet, COCO
✓ "supervised learning", "supervised training" explicit mention

SUPERVISED EVEN WITHOUT SAYING "SUPERVISED":
✓ Training on SAT/UNSAT labeled instances → SUPERVISED
✓ Training on theorem provability labels → SUPERVISED  
✓ Training on graph property labels → SUPERVISED
✓ Using CIFAR-10 for classification → SUPERVISED (it's labeled!)
✓ Training with "correct/incorrect" labels → SUPERVISED
✓ Evaluation with accuracy/F1 → implies SUPERVISED

═══ STEP 4: REASONING CHECKLIST ═══

Ask yourself:
1. Does training use INPUT-OUTPUT PAIRS where OUTPUT is KNOWN?
   - Yes → Likely SUPERVISED
   - No → Check other paradigms

2. Are the outputs HUMAN-ANNOTATED or GROUND TRUTH?
   - Yes → SUPERVISED
   - Auto-generated/self-created → NOT supervised

3. What's the loss function?
   - Requires labels (cross-entropy, classification) → SUPERVISED
   - Reconstruction, contrastive → NOT supervised
   
4. What are the evaluation metrics?
   - Accuracy, F1, precision/recall → SUPERVISED
   - Clustering metrics, reconstruction error → NOT supervised

5. What's the dataset?
   - CIFAR/MNIST/ImageNet → LABELED → SUPERVISED
   - Unlabeled image collection → NOT supervised

═══ EXAMPLES WITH REASONING ═══

EXAMPLE 1: "We train a GNN on SAT instances from SatLib benchmark"
→ REASONING: SAT instances have KNOWN satisfiability (SAT/UNSAT labels)
→ Training on labeled SAT instances
→ ANSWER: YES (supervised)

EXAMPLE 2: "We use contrastive learning on ImageNet"
→ REASONING: Contrastive creates own pretext task (augmentation pairs)
→ Not using ImageNet's human labels, using self-supervision
→ ANSWER: NO (self-supervised)

EXAMPLE 3: "We train on CIFAR-10 for image classification"
→ REASONING: CIFAR-10 has 10 CLASS LABELS (airplane, car, etc.)
→ Classification task uses these GROUND TRUTH labels
→ ANSWER: YES (supervised)

EXAMPLE 4: "We cluster protein structures to find patterns"
→ REASONING: Clustering has no labels, finds structure in data
→ ANSWER: NO (unsupervised)

EXAMPLE 5: "We predict graph properties from labeled training data"
→ REASONING: Explicitly mentions LABELED training data
→ Prediction task with known targets
→ ANSWER: YES (supervised)

═══ YOUR TASK ═══
Read the content below. Apply the reasoning framework above.
Think step-by-step: What paradigm? Are there labels? What's the task?

Relevant content:
{all_content['supervised'][:1200]}

═══════════════════════════════════════════════════════════════════════════════
CRITERION 3: DATA EFFICIENCY - MULTI-STEP REASONING
═══════════════════════════════════════════════════════════════════════════════

CORE QUESTION: Can this method be trained with MODEST/ACCESSIBLE amounts of data?
This is about PRACTICAL DATA REQUIREMENTS, not strict cutoffs.

═══ REASONING FRAMEWORK ═══

STEP 1: Understand the domain and what "small" means for that domain

DIFFERENT DOMAINS HAVE DIFFERENT SCALES:
- Computer Vision: 
  * Small: CIFAR (60K), MNIST (70K), small ImageNet subsets
  * Medium: Full ImageNet-1K (~1M) - ACCEPTABLE if method works here
  * Large: ImageNet-21K (14M), web-scale datasets
  
- NLP/Text:
  * Small: Small corpora, specific domain texts (10K-100K samples)
  * Medium: WikiText, BookCorpus subsets
  * Large: Full web scrapes, billion-token datasets
  
- Graphs/SAT/Combinatorial:
  * Small: Hundreds to thousands of instances
  * Medium: 10K-100K problem instances - ACCEPTABLE
  * Large: Million-scale synthetic datasets
  
- Tabular/Scientific:
  * Small: Dozens to hundreds of samples (common in science!)
  * Medium: Thousands of samples
  * Large: Million-row datasets

STEP 2: What did the paper actually use?

INDICATORS OF DATA EFFICIENCY (Answer: YES):
✓ Explicit small datasets: CIFAR-10/100, MNIST, Fashion-MNIST, SVHN
✓ Domain-appropriate modest sizes: 
  * 1K-100K instances for graphs/SAT/structured problems
  * <1M images for vision
  * <1M sentences for NLP
✓ Papers emphasizing: "data-efficient", "limited data", "few-shot"
✓ Papers working with: "small benchmarks", "standard benchmarks"
✓ Method designed for: "low-data regimes", "data scarcity"
✓ Successfully trains on: publicly available, downloadable datasets
✓ Real-world constraints: medical imaging (limited samples), scientific data

INDICATORS OF REQUIRING MASSIVE DATA (Answer: NO):
✗ Requires: ImageNet-21K, JFT-300M, billion-scale datasets
✗ States: "requires large-scale pre-training"
✗ Uses: web-scale scraping, "collected millions of images"
✗ Needs: "billion-token corpus", "entire web crawl"
✗ Method breaks down: "without sufficient data (>1M samples)"

STEP 3: Consider the METHOD's data hunger

DATA-EFFICIENT METHODS (Lean toward YES):
✓ Methods that work well on standard benchmarks
✓ Techniques with: regularization, data augmentation, transfer learning
✓ Architectures: compact models, efficient designs
✓ Claims about: sample efficiency, working with limited data

DATA-HUNGRY METHODS (Lean toward NO):
✗ "Scales with data" as key selling point
✗ Pre-training phases requiring massive datasets
✗ Performance critically depends on data scale
✗ Method specifically designed for big data scenarios

STEP 4: Practical accessibility

ACCESSIBLE DATA (Answer: YES):
✓ Can download in minutes/hours
✓ Fits on consumer hardware (single disk, modest RAM)
✓ Standard research benchmarks
✓ Publicly available datasets
✓ Reasonable to collect (surveys, experiments)

INACCESSIBLE DATA (Answer: NO):
✗ Requires: web-scale infrastructure, data centers
✗ Proprietary: "collected from our production system"
✗ Unreasonably expensive: "annotated by 1000 workers over 2 years"

═══ EXAMPLES WITH REASONING ═══

EXAMPLE 1: "We evaluate on ImageNet-1K achieving 75% accuracy"
→ REASONING: ImageNet-1K is 1.2M images, large but STANDARD benchmark
→ Many methods work here, it's accessible, downloadable
→ Not extreme scale, commonly used in research
→ ANSWER: YES (acceptable data scale for vision)

EXAMPLE 2: "Pre-trained on JFT-300M, fine-tuned on ImageNet"
→ REASONING: JFT-300M is 300 million images, massive proprietary dataset
→ Requires enormous infrastructure
→ Method requires massive pre-training
→ ANSWER: NO (requires inaccessible massive data)

EXAMPLE 3: "Tested on 5K SAT instances from competition benchmarks"
→ REASONING: For SAT domain, 5K instances is reasonable
→ Publicly available competition data
→ Modest scale appropriate for the problem
→ ANSWER: YES (good data efficiency for domain)

EXAMPLE 4: "Trained on CIFAR-10 with standard augmentation"
→ REASONING: CIFAR-10 is 60K images, classic small dataset
→ Very accessible, fits on any computer
→ ANSWER: YES (clearly data-efficient)

EXAMPLE 5: "Method requires large-scale self-supervised pre-training"
→ REASONING: "Large-scale" implies millions/billions of samples
→ Self-supervised often means web-scale data
→ Practical accessibility questionable
→ ANSWER: NO or UNKNOWN (data-hungry approach)

═══ KEY PRINCIPLE ═══

Answer YES if:
- Method works with ACCESSIBLE, DOWNLOADABLE, REASONABLE datasets
- Data scale is APPROPRIATE for the domain
- NOT requiring web-scale, billion-sample, proprietary megadatasets
- A grad student could reasonably obtain and use this data

Answer NO if:
- Requires MASSIVE proprietary datasets
- Needs infrastructure beyond academic research labs
- Web-scale data requirements
- Extreme data hunger is core to method

Relevant content:
{all_content['small_dataset'][:1200]}

═══════════════════════════════════════════════════════════════════════════════
CRITERION 4: PRACTICAL TRAINING FEASIBILITY - MULTI-STEP REASONING  
═══════════════════════════════════════════════════════════════════════════════

CORE QUESTION: Can this method be trained with ACCESSIBLE COMPUTATIONAL RESOURCES?
This is about REPRODUCIBILITY and PRACTICAL FEASIBILITY, not strict time limits.

═══ REASONING FRAMEWORK ═══

STEP 1: Understand what "accessible" means

ACCESSIBLE COMPUTE = What a typical research lab has:
- Single consumer/prosumer GPU (RTX 3090, A100, V100)
- Reasonable training time (hours to couple days, not weeks)
- Doesn't require: massive clusters, TPU farms, supercomputers

INACCESSIBLE COMPUTE = Industry/big-lab scale:
- Multi-node GPU clusters (8+ GPUs, multiple machines)
- TPU pods (multiple TPU chips coordinated)
- Training measured in weeks or months
- Requires: specialized infrastructure, huge power/cooling

STEP 2: Analyze hardware requirements

FEASIBLE HARDWARE (Answer: YES):
✓ "Single GPU", "one V100", "RTX 3090"
✓ "Trained on a single machine"
✓ No distributed training mentioned
✓ Modest hardware requirements
✓ Consumer/prosumer GPUs sufficient
✓ "Easily reproducible", "can be run on standard hardware"

INFEASIBLE HARDWARE (Answer: NO):
✗ "64 GPUs", "8-node cluster", "DGX station"
✗ "TPU v3 pods", "TPU v4", "Cloud TPU"
✗ "Requires distributed training"
✗ "Trained on supercomputer"
✗ Multiple machines coordinated

STEP 3: Consider training time relative to domain

DOMAIN-APPROPRIATE TIMES:

Vision (images/video):
- Small models, small data: Hours - FEASIBLE
- Medium models (ResNet-50 on ImageNet): 1-3 days - ACCEPTABLE  
- Large models (ViT-Large, multi-week): - INFEASIBLE

NLP/Language:
- Small models on modest data: Hours to days - FEASIBLE
- BERT-base scale: Few days - ACCEPTABLE
- GPT-3 scale: Weeks/months - INFEASIBLE

Graphs/SAT/Combinatorial:
- GNNs on thousands of graphs: Hours - FEASIBLE
- Large-scale graph models: Days - ACCEPTABLE
- Extreme scale: Weeks - INFEASIBLE

General heuristic:
- Hours: Clearly feasible ✓
- 1-2 days: Acceptable for research ✓
- 3-7 days: Borderline, acceptable if necessary ~
- Weeks+: Generally infeasible ✗

STEP 4: Analyze model complexity

PRACTICAL MODELS (Lean toward YES):
✓ Parameters: <100M (very practical)
✓ Parameters: 100M-500M (practical on modern GPUs)
✓ Efficient architectures: MobileNet, EfficientNet, compact transformers
✓ Methods emphasizing: efficiency, speed, compactness
✓ Pruning, quantization, distillation techniques

IMPRACTICAL MODELS (Lean toward NO):
✗ Parameters: >1B (billion-scale)
✗ Very deep/wide networks requiring huge memory
✗ Ensemble of multiple large models
✗ "Requires significant resources"

STEP 5: Consider practical indicators

REPRODUCIBLE (Answer: YES):
✓ Clear hardware specifications matching academic resources
✓ Training time explicitly stated and reasonable
✓ "Experiments run on single GPU"
✓ Method designed for efficiency
✓ Fast convergence, early stopping effective
✓ Standard optimization, no exotic requirements

NOT REPRODUCIBLE (Answer: NO):
✗ Vague: "trained on our cluster" (how big?)
✗ Extreme scale explicitly mentioned
✗ "Computationally intensive"
✗ "Requires substantial resources"
✗ Infrastructure clearly beyond typical labs

═══ EXAMPLES WITH REASONING ═══

EXAMPLE 1: "Trained ResNet-50 on ImageNet for 90 epochs on 8 V100 GPUs (3 days)"
→ REASONING: 8 GPUs is multi-GPU but not extreme
→ 3 days is reasonable for ImageNet
→ V100s are research-grade GPUs (accessible)
→ Standard ImageNet training setup
→ ANSWER: YES (feasible for well-equipped lab)

EXAMPLE 2: "Training takes ~12 hours on a single RTX 3090"
→ REASONING: Consumer GPU, explicit short time
→ Clearly reproducible
→ ANSWER: YES (very feasible)

EXAMPLE 3: "Pre-trained on 64 TPU v3 chips for 2 weeks"
→ REASONING: 64 TPUs is massive infrastructure
→ 2 weeks is very long
→ TPU pods not accessible to most researchers
→ ANSWER: NO (requires industry resources)

EXAMPLE 4: "Small GNN model, trains in 2 hours on single GPU"
→ REASONING: Explicit single GPU, very short time
→ Small model, efficient
→ ANSWER: YES (highly feasible)

EXAMPLE 5: "Trained for 1 week on 4 GPUs"
→ REASONING: 1 week is long but not extreme
→ 4 GPUs is multi-GPU but modest scale
→ Borderline but within reach of research labs
→ ANSWER: YES or UNKNOWN (borderline acceptable)

EXAMPLE 6: "Requires significant computational resources for training"
→ REASONING: Vague but negative signal
→ "Significant" suggests beyond normal
→ No specific details to judge
→ ANSWER: UNKNOWN or NO (likely infeasible)

═══ KEY PRINCIPLE ═══

Answer YES if:
- Can be trained with TYPICAL ACADEMIC LAB resources
- Single GPU or modest multi-GPU (2-4 GPUs)
- Training time: hours to few days
- Model size: practical for modern GPUs
- Method explicitly designed for efficiency OR uses standard resources

Answer NO if:
- Requires: massive clusters, TPU farms, supercomputers
- Training time: weeks or months
- Infrastructure: clearly beyond academic research
- Explicitly states: "substantial resources", "large-scale infrastructure"

Answer UNKNOWN if:
- No hardware/time details provided
- Ambiguous resource requirements
- Can't determine from content

Relevant content:
{all_content['quick_training'][:1200]}

═══════════════════════════════════════════════════════════════════════════════
CRITERION 5: CODE AVAILABILITY - MULTI-STEP REASONING
═══════════════════════════════════════════════════════════════════════════════

CORE QUESTION: Is code PUBLICLY ACCESSIBLE for reproduction?

═══ REASONING FRAMEWORK ═══

STEP 1: Look for explicit availability

CLEARLY AVAILABLE NOW (Answer: YES):
✓ GitHub/GitLab URL: "github.com/user/repo", "gitlab.com/user/project"
✓ "Code available at [URL]"
✓ "Implementation: https://..."
✓ "Open source at [URL]"
✓ "Our code: [URL]"
✓ "See our repository at [URL]"

CLEARLY NOT AVAILABLE (Answer: NO):
✗ "Code will be released upon publication" (FUTURE)
✗ "Available upon acceptance/publication" (FUTURE)
✗ "Code available upon request" (NOT public, requires emailing)
✗ "Will be released after review" (FUTURE)
✗ "Coming soon" (FUTURE)
✗ No code availability mentioned at all

STEP 2: Infer from context

LIKELY HAS CODE (Lean toward YES):
✓ "Open source implementation"
✓ "Publicly available"
✓ "We release our code" (past tense suggests done)
✓ "Code is available" (present tense, not future)
✓ Implementation details so specific it implies shared code
✓ References to "our GitHub repository"

LIKELY NO CODE (Lean toward NO):
✗ Industry/proprietary context
✗ No implementation details at all
✗ Theoretical focus with no mention of code
✗ "Proprietary implementation"

STEP 3: Consider practical reproducibility

REPRODUCIBLE WITHOUT CODE (Still answer based on explicit availability):
- Even if method is simple and reproducible from description
- Answer based on whether code is ACTUALLY SHARED
- Don't assume availability without evidence

COMPLEX BUT NO CODE (Answer: NO):
- Complex methods needing code for reproduction
- But no code explicitly available
- Still answer NO if not available

═══ EXAMPLES WITH REASONING ═══

EXAMPLE 1: "Code: github.com/user/project"
→ REASONING: Explicit GitHub URL
→ ANSWER: YES (clearly available)

EXAMPLE 2: "Implementation will be released upon publication"
→ REASONING: Future promise, not available NOW
→ ANSWER: NO (not currently available)

EXAMPLE 3: "Code available upon request"
→ REASONING: Not PUBLIC, requires personal request
→ ANSWER: NO (not publicly available)

EXAMPLE 4: "Our open source implementation is available online"
→ REASONING: "Is available" (present tense), "open source"
→ Implies currently accessible
→ ANSWER: YES or UNKNOWN (likely yes, but no URL)

EXAMPLE 5: No mention of code at all
→ REASONING: Nothing stated about availability
→ Can't assume availability
→ ANSWER: NO (absence of evidence)

EXAMPLE 6: "We provide a PyTorch implementation in our repository"
→ REASONING: "Provide" and "repository" suggest available
→ Present tense implies current availability
→ ANSWER: YES (contextual evidence of availability)

═══ KEY PRINCIPLE ═══

Answer YES if:
- Explicit URL to public repository (GitHub, GitLab, etc.)
- Clear statement of current public availability
- Present tense: "code is available", "we provide code"

Answer NO if:
- Future promises: "will release", "upon publication"
- Non-public: "upon request", "contact us"
- No mention of code availability at all
- Proprietary/closed implementations

Answer UNKNOWN if:
- Ambiguous statements
- Contextual hints but no explicit confirmation
- Can't determine from content

═══ SPECIAL NOTE ═══

Don't be generous without evidence. "NO code mentioned" = NO, not UNKNOWN.
Only answer YES if there's EXPLICIT or VERY STRONG evidence of public availability.

Relevant content:
{all_content['has_repo'][:1200]}

================================================================================
OUTPUT FORMAT - RESPOND WITH DETAILED REASONING
================================================================================

Based on your MULTI-STEP REASONING above, respond with ONLY this JSON:

{{
    "pytorch": {{
        "answer": "Yes|No|Unknown",
        "confidence": 0.0-1.0,
        "evidence": "What you found - explicit mention or contextual inference"
    }},
    "supervised": {{
        "answer": "Yes|No|Unknown",
        "confidence": 0.0-1.0,
        "evidence": "YOUR REASONING: What paradigm? What labels? Why supervised or not?"
    }},
    "small_dataset": {{
        "answer": "Yes|No|Unknown",
        "confidence": 0.0-1.0,
        "evidence": "Dataset name/size, domain context, accessibility reasoning"
    }},
    "quick_training": {{
        "answer": "Yes|No|Unknown",
        "confidence": 0.0-1.0,
        "evidence": "Hardware, time, model size, feasibility reasoning"
    }},
    "has_repo": {{
        "answer": "Yes|No|Unknown",
        "confidence": 0.0-1.0,
        "evidence": "URL or availability statement with present/future tense noted"
    }}
}}

REMEMBER: Use the MULTI-STEP REASONING FRAMEWORKS above. Don't just keyword match!"""
        
        # MAKE ONE BEDROCK CALL FOR ALL CRITERIA
        try:
            response = self._call_bedrock(mega_prompt)
            
            # Parse JSON response with all criteria
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                # Convert to CriterionResult for each criterion
                evaluations = {}
                for criterion_name in self.criteria.keys():
                    if criterion_name in data:
                        crit_data = data[criterion_name]
                        evaluations[criterion_name] = CriterionResult(
                            name=criterion_name,
                            answer=crit_data.get("answer", "Unknown"),
                            confidence=float(crit_data.get("confidence", 0.5)),
                            evidence=crit_data.get("evidence", ""),
                            sections=[]
                        )
                        
                        # Update status with this criterion
                        with self.status_lock:
                            if doc_id in self.paper_statuses:
                                self.paper_statuses[doc_id].criteria_results[criterion_name] = crit_data.get("answer", "Unknown")
                    else:
                        # Fallback if criterion missing
                        evaluations[criterion_name] = CriterionResult(
                            name=criterion_name,
                            answer="Unknown",
                            confidence=0.3,
                            evidence="Not in response",
                            sections=[]
                        )
            else:
                # Fallback if no JSON found
                print(f"⚠️ No valid JSON in response, using fallback")
                evaluations = {
                    name: CriterionResult(name=name, answer="Unknown", confidence=0.3, evidence="Parse error", sections=[])
                    for name in self.criteria.keys()
                }
        
        except Exception as e:
            print(f"❌ Error in bulk evaluation: {e}")
            evaluations = {
                name: CriterionResult(name=name, answer="Unknown", confidence=0.0, evidence=f"Error: {e}", sections=[])
                for name in self.criteria.keys()
            }
        
        # Calculate final score
        overall_score = self._calculate_score(evaluations)
        recommendation = self._make_recommendation(evaluations, overall_score)
        
        # Update final status
        with self.status_lock:
            if doc_id in self.paper_statuses:
                self.paper_statuses[doc_id].status = "completed"
                self.paper_statuses[doc_id].progress = 100
                self.paper_statuses[doc_id].overall_score = overall_score
                self.paper_statuses[doc_id].recommendation = recommendation
        
        if update_callback:
            update_callback(doc_id, "completed", 100)
        
        print(f"✅ {title}: {recommendation} ({overall_score:.2f})")
        
        return PaperResult(
            doc_id=doc_id,
            title=title,
            evaluations=evaluations,
            overall_score=overall_score,
            recommendation=recommendation,
            summary=f"{title} - {recommendation} (Score: {overall_score:.2f})"
        )
    
    def analyze_paper_separate_criteria(self, doc_id: str, update_callback=None) -> Optional[PaperResult]:
        """Analyze single paper with SEPARATE API calls for each criterion (5 calls per paper)"""
        # Get document
        doc_meta = self.document_db.get_document(doc_id)
        if not doc_meta:
            return None
        
        sections = self.document_db.get_document_sections(doc_id)
        if not sections:
            return None
        
        title = doc_meta['title']
        
        # Initialize status
        with self.status_lock:
            self.paper_statuses[doc_id] = PaperStatus(
                doc_id=doc_id,
                title=title,
                status="running",
                progress=0,
                criteria_results={},
                overall_score=0.0,
                recommendation="Pending"
            )
        
        if update_callback:
            update_callback(doc_id, "running", 0)
        
        print(f"🔍 Analyzing (SEPARATE): {title}")
        
        # Evaluate EACH criterion separately (5 API calls)
        evaluations = {}
        criteria_list = list(self.criteria.keys())
        total_criteria = len(criteria_list)
        
        for idx, criterion_name in enumerate(criteria_list):
            # Get relevant content for this specific criterion
            relevant_sections = self.criteria_analyzer.get_section_content_for_criteria(
                doc_id, criterion_name, top_k=2
            )
            
            content_parts = []
            for section_info in relevant_sections:
                key = section_info['section_key']
                if key in sections:
                    content_parts.append(f"{sections[key][:500]}")
            
            content = "\n".join(content_parts) if content_parts else str(sections)[:1000]
            
            # Evaluate THIS criterion with separate API call
            print(f"  📊 Criterion {idx+1}/{total_criteria}: {criterion_name}")
            result = self._evaluate_criterion(doc_id, criterion_name, content)
            evaluations[criterion_name] = result
            
            # Update status after each criterion
            progress = int(((idx + 1) / total_criteria) * 100)
            with self.status_lock:
                if doc_id in self.paper_statuses:
                    self.paper_statuses[doc_id].criteria_results[criterion_name] = result.answer
                    self.paper_statuses[doc_id].progress = progress
            
            if update_callback:
                update_callback(doc_id, "running", progress)
        
        # Calculate final score
        overall_score = self._calculate_score(evaluations)
        recommendation = self._make_recommendation(evaluations, overall_score)
        
        # Update final status
        with self.status_lock:
            if doc_id in self.paper_statuses:
                self.paper_statuses[doc_id].status = "completed"
                self.paper_statuses[doc_id].progress = 100
                self.paper_statuses[doc_id].overall_score = overall_score
                self.paper_statuses[doc_id].recommendation = recommendation
        
        if update_callback:
            update_callback(doc_id, "completed", 100)
        
        print(f"✅ {title}: {recommendation} ({overall_score:.2f})")
        
        return PaperResult(
            doc_id=doc_id,
            title=title,
            evaluations=evaluations,
            overall_score=overall_score,
            recommendation=recommendation,
            summary=f"{title} - {recommendation} (Score: {overall_score:.2f})"
        )
    
    def analyze_all_papers_separate_criteria(self, update_callback=None) -> List[PaperResult]:
        """Analyze all papers with SEPARATE criterion evaluation (5 API calls per paper)"""
        documents = self.document_db.list_documents()
        
        # Initialize all statuses
        with self.status_lock:
            for doc_info in documents:
                doc_id = doc_info['doc_id']
                self.paper_statuses[doc_id] = PaperStatus(
                    doc_id=doc_id,
                    title=doc_info['title'],
                    status="pending",
                    progress=0,
                    criteria_results={},
                    overall_score=0.0,
                    recommendation="Pending"
                )
        
        print(f"\n🔬 Starting SEPARATE criteria analysis of {len(documents)} papers...")
        print(f"   Total API calls: {len(documents) * 5}")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all papers for analysis
            future_to_doc = {
                executor.submit(self.analyze_paper_separate_criteria, doc_info['doc_id'], update_callback): doc_info
                for doc_info in documents
            }
            
            # Process as they complete
            for future in concurrent.futures.as_completed(future_to_doc):
                doc_info = future_to_doc[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"❌ Error analyzing {doc_info['title']}: {e}")
                    # Mark as failed
                    with self.status_lock:
                        if doc_info['doc_id'] in self.paper_statuses:
                            self.paper_statuses[doc_info['doc_id']].status = "failed"
                            self.paper_statuses[doc_info['doc_id']].error = str(e)
        
        print(f"\n✅ Analysis complete! {len(results)}/{len(documents)} papers analyzed")
        return results
    
    def analyze_all_papers_parallel(self, update_callback=None) -> List[PaperResult]:
        """Analyze all papers in parallel (ONE mega-prompt per paper)"""
        documents = self.document_db.list_documents()
        
        # Initialize all statuses
        with self.status_lock:
            for doc_info in documents:
                doc_id = doc_info['doc_id']
                self.paper_statuses[doc_id] = PaperStatus(
                    doc_id=doc_id,
                    title=doc_info['title'],
                    status="pending",
                    progress=0,
                    criteria_results={},
                    overall_score=0.0,
                    recommendation="Pending"
                )
        
        print(f"\n🚀 Starting parallel analysis of {len(documents)} papers with {self.max_workers} workers...")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all papers for analysis
            future_to_doc = {
                executor.submit(self.analyze_paper, doc_info['doc_id'], update_callback): doc_info
                for doc_info in documents
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_doc):
                doc_info = future_to_doc[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"❌ Error analyzing {doc_info['title']}: {e}")
                    with self.status_lock:
                        if doc_info['doc_id'] in self.paper_statuses:
                            self.paper_statuses[doc_info['doc_id']].status = "failed"
                            self.paper_statuses[doc_info['doc_id']].error = str(e)
        
        print(f"\n✅ Parallel analysis complete! {len(results)}/{len(documents)} papers analyzed")
        
        return results
    
    def get_all_statuses(self) -> List[PaperStatus]:
        """Get all paper statuses for UI"""
        with self.status_lock:
            return list(self.paper_statuses.values())
    
    def get_status(self, doc_id: str) -> Optional[PaperStatus]:
        """Get status for specific paper"""
        with self.status_lock:
            return self.paper_statuses.get(doc_id)
    
    def _calculate_score(self, evaluations: Dict[str, CriterionResult]) -> float:
        """Calculate weighted score"""
        total_weight = 0.0
        weighted_score = 0.0
        
        for criterion_name, evaluation in evaluations.items():
            criterion_def = self.criteria[criterion_name]
            weight = criterion_def['weight']
            total_weight += weight
            
            if evaluation.answer == "Yes":
                score = 1.0 * evaluation.confidence
            elif evaluation.answer == "No":
                score = 0.0
            else:
                score = 0.5 * (1 - evaluation.confidence)
            
            weighted_score += score * weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _make_recommendation(self, evaluations: Dict[str, CriterionResult], overall_score: float) -> str:
        """Make recommendation"""
        # Check required criteria
        for criterion_name, evaluation in evaluations.items():
            if self.criteria[criterion_name]['required']:
                if evaluation.answer == "No" and evaluation.confidence > 0.7:
                    return "Exclude"
        
        if overall_score >= 0.8:
            return "Include"
        elif overall_score >= 0.5:
            return "Review"
        else:
            return "Exclude"
    
    def save_results(self, results: List[PaperResult], output_file: Optional[Path] = None):
        """Save results to JSON"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"bedrock_analysis_{timestamp}.json")
        
        data = {
            'results': [r.to_dict() for r in results],
            'summary': {
                'total': len(results),
                'include': sum(1 for r in results if r.recommendation == "Include"),
                'exclude': sum(1 for r in results if r.recommendation == "Exclude"),
                'review': sum(1 for r in results if r.recommendation == "Review"),
                'method': 'Bedrock-Parallel'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")

