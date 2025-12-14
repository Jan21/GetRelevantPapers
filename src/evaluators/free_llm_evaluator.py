#!/usr/bin/env python3
"""
FREE LLM Paper Evaluator using Ollama (Local Models)
No API keys, no authentication, completely FREE!
"""

import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.markdown_parser import DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer
from src.core.analyzer import CriterionResult, PaperResult


class OllamaClient:
    """Client for Ollama local LLM models - completely FREE!"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2:3b"  # Fast, free, local model
        
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def install_model(self, model_name: str = None) -> bool:
        """Install a model if not already available"""
        model_name = model_name or self.model
        try:
            print(f"📥 Installing {model_name} (this may take a few minutes)...")
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Failed to install model: {e}")
            return False
    
    def chat_completion(self, messages: List[Dict], model: Optional[str] = None) -> str:
        """Make chat completion request to Ollama"""
        model = model or self.model
        
        # Convert messages to single prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 500
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"Ollama API error {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Ollama request failed: {str(e)}")


class FreeLLMPaperEvaluator:
    """Paper evaluator using FREE local LLM models via Ollama"""
    
    def __init__(self, document_db: DocumentDatabase, vector_store: SimpleVectorStore):
        self.document_db = document_db
        self.vector_store = vector_store
        self.criteria_analyzer = CriteriaAnalyzer(vector_store)
        self.ollama_client = OllamaClient()
        
        # Define evaluation criteria with prompts
        self.criteria = {
            'pytorch': {
                'system_prompt': "You are an expert at analyzing research papers for PyTorch usage.",
                'evaluation_prompt': """Analyze this paper content and determine if it uses PyTorch.

Paper content:
{content}

Look for:
- Mentions of "pytorch", "torch", "torchvision"
- PyTorch-specific code or imports
- References to PyTorch in implementation

Answer with ONLY one word: "Yes", "No", or "Unknown"
Then on a new line, provide confidence (0.0-1.0)
Then on a new line, provide brief evidence (max 50 words)

Format:
Answer: [Yes/No/Unknown]
Confidence: [0.0-1.0]
Evidence: [brief explanation]"""
            },
            'supervised': {
                'system_prompt': "You are an expert at analyzing research papers for supervised learning approaches.",
                'evaluation_prompt': """Analyze this paper content and determine if it uses supervised learning.

Paper content:
{content}

Look for:
- Mentions of "supervised", "classification", "labeled data"
- Training with ground truth labels
- Cross-entropy loss, accuracy metrics

Answer with ONLY one word: "Yes", "No", or "Unknown"
Then on a new line, provide confidence (0.0-1.0)
Then on a new line, provide brief evidence (max 50 words)

Format:
Answer: [Yes/No/Unknown]
Confidence: [0.0-1.0]
Evidence: [brief explanation]"""
            },
            'small_dataset': {
                'system_prompt': "You are an expert at analyzing research papers for dataset size.",
                'evaluation_prompt': """Analyze this paper content and determine if it uses small datasets.

Paper content:
{content}

Look for:
- Small datasets like CIFAR, MNIST
- Mentions of "small-scale", "limited data", "few-shot"
- Dataset sizes under 100K samples

Answer with ONLY one word: "Yes", "No", or "Unknown"
Then on a new line, provide confidence (0.0-1.0)
Then on a new line, provide brief evidence (max 50 words)

Format:
Answer: [Yes/No/Unknown]
Confidence: [0.0-1.0]
Evidence: [brief explanation]"""
            },
            'quick_training': {
                'system_prompt': "You are an expert at analyzing research papers for training efficiency.",
                'evaluation_prompt': """Analyze this paper content and determine if it has quick/efficient training.

Paper content:
{content}

Look for:
- Mentions of "single GPU", "hours", "fast training"
- Efficient architectures, lightweight models
- Quick convergence, fast training times

Answer with ONLY one word: "Yes", "No", or "Unknown"
Then on a new line, provide confidence (0.0-1.0)
Then on a new line, provide brief evidence (max 50 words)

Format:
Answer: [Yes/No/Unknown]
Confidence: [0.0-1.0]
Evidence: [brief explanation]"""
            },
            'has_repo': {
                'system_prompt': "You are an expert at analyzing research papers for code availability.",
                'evaluation_prompt': """Analyze this paper content and determine if code/repository is available.

Paper content:
{content}

Look for:
- GitHub/GitLab URLs
- Mentions of "code available", "implementation available"
- Repository links, open source mentions

Answer with ONLY one word: "Yes", "No", or "Unknown"
Then on a new line, provide confidence (0.0-1.0)
Then on a new line, provide brief evidence (max 50 words)

Format:
Answer: [Yes/No/Unknown]
Confidence: [0.0-1.0]
Evidence: [brief explanation]"""
            }
        }
    
    def setup_ollama(self) -> bool:
        """Setup Ollama and install required model"""
        print("🔧 Setting up FREE local LLM with Ollama...")
        
        # Check if Ollama is running
        if not self.ollama_client.is_available():
            print("❌ Ollama is not running!")
            print("📥 Please install and start Ollama:")
            print("   1. Download from: https://ollama.ai/")
            print("   2. Install and run: ollama serve")
            print("   3. Then run this script again")
            return False
        
        print("✅ Ollama is running!")
        
        # Install model if needed
        try:
            # Test if model is available
            test_response = self.ollama_client.chat_completion([
                {"role": "user", "content": "Hello"}
            ])
            print(f"✅ Model {self.ollama_client.model} is ready!")
            return True
        except:
            print(f"📥 Installing model {self.ollama_client.model}...")
            if self.ollama_client.install_model():
                print("✅ Model installed successfully!")
                return True
            else:
                print("❌ Failed to install model")
                return False
    
    def evaluate_criterion(self, doc_id: str, criterion_name: str, content: str) -> CriterionResult:
        """Evaluate a single criterion using FREE local LLM"""
        try:
            criterion_def = self.criteria[criterion_name]
            
            messages = [
                {"role": "system", "content": criterion_def['system_prompt']},
                {"role": "user", "content": criterion_def['evaluation_prompt'].format(content=content)}
            ]
            
            response = self.ollama_client.chat_completion(messages)
            
            # Parse response
            lines = response.strip().split('\n')
            answer = "Unknown"
            confidence = 0.0
            evidence = "No response from LLM"
            
            for line in lines:
                line = line.strip()
                if line.startswith("Answer:"):
                    answer = line.split(":", 1)[1].strip()
                elif line.startswith("Confidence:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith("Evidence:"):
                    evidence = line.split(":", 1)[1].strip()
            
            return CriterionResult(
                name=criterion_name,
                answer=answer,
                confidence=confidence,
                evidence=evidence,
                sections=[]
            )
            
        except Exception as e:
            print(f"❌ FREE LLM evaluation failed for {criterion_name}: {e}")
            return CriterionResult(
                name=criterion_name,
                answer="Unknown",
                confidence=0.0,
                evidence=f"FREE LLM evaluation failed: {str(e)}",
                sections=[]
            )
    
    def analyze_paper(self, doc_id: str) -> Optional[PaperResult]:
        """Analyze a paper using FREE local LLM evaluation"""
        # Get document metadata and sections
        doc_meta = self.document_db.get_document(doc_id)
        if not doc_meta:
            print(f"Document {doc_id} not found")
            return None
        
        sections = self.document_db.get_document_sections(doc_id)
        if not sections:
            print(f"No sections found for document {doc_id}")
            return None
        
        print(f"Analyzing with FREE LLM: {doc_meta['title']}")
        
        # Get relevant sections for analysis
        relevant_sections = self.criteria_analyzer.find_relevant_sections(
            doc_id, 
            ["pytorch", "supervised", "small_dataset", "quick_training", "has_repo"]
        )
        
        # Combine relevant content (limit to prevent overwhelming the LLM)
        content_parts = []
        for section_name in relevant_sections[:5]:  # Top 5 sections
            if section_name in sections:
                content_parts.append(f"## {section_name}\n{sections[section_name]}")
        
        content = "\n\n".join(content_parts)[:3000]  # Limit to 3000 chars
        
        # Analyze each criterion
        evaluations = {}
        for criterion_name in self.criteria.keys():
            print(f"  🔍 Evaluating {criterion_name}...")
            result = self.evaluate_criterion(doc_id, criterion_name, content)
            evaluations[criterion_name] = result
        
        # Calculate overall score
        total_score = 0
        total_weight = 0
        
        criterion_weights = {
            'pytorch': 1.0,
            'supervised': 1.0, 
            'small_dataset': 0.6,
            'quick_training': 0.4,
            'has_repo': 1.0
        }
        
        for criterion_name, result in evaluations.items():
            weight = criterion_weights.get(criterion_name, 1.0)
            if result.answer.lower() == "yes":
                score = result.confidence
            elif result.answer.lower() == "no":
                score = 0.0
            else:  # Unknown
                score = 0.5
            
            total_score += score * weight
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.5
        
        # Determine recommendation
        if overall_score >= 0.8:
            recommendation = "Include"
        elif overall_score >= 0.5:
            recommendation = "Review"
        else:
            recommendation = "Exclude"
        
        summary = f"{doc_meta['title']} - {recommendation} (FREE LLM Score: {overall_score:.2f})"
        
        return PaperResult(
            doc_id=doc_id,
            title=doc_meta['title'],
            evaluations=evaluations,
            overall_score=overall_score,
            recommendation=recommendation,
            summary=summary
        )
    
    def save_results(self, results: List[PaperResult], output_file: Optional[Path] = None):
        """Save results to JSON (compatible with other analyzers)"""
        if output_file is None:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"free_llm_analysis_{timestamp}.json")
        
        data = {
            'results': [r.to_dict() for r in results],
            'summary': {
                'total': len(results),
                'include': sum(1 for r in results if r.recommendation == "Include"),
                'exclude': sum(1 for r in results if r.recommendation == "Exclude"),
                'review': sum(1 for r in results if r.recommendation == "Review"),
                'method': 'FREE_LLM'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 FREE LLM results saved to: {output_file}")


def main():
    """Test the FREE LLM evaluator"""
    from pathlib import Path
    
    # Setup
    db_path = Path("markdown_db")
    vector_path = Path("vector_store")
    
    db = DocumentDatabase(db_path)
    vector_store = SimpleVectorStore(vector_path)
    
    evaluator = FreeLLMPaperEvaluator(db, vector_store)
    
    # Setup Ollama
    if not evaluator.setup_ollama():
        print("❌ Failed to setup Ollama")
        return
    
    # Test on first document
    docs = db.list_documents()
    if docs:
        doc_info = docs[0]
        print(f"\n🧪 Testing FREE LLM on: {doc_info['title']}")
        
        result = evaluator.analyze_paper(doc_info['doc_id'])
        if result:
            print(f"\n✅ Analysis complete!")
            print(f"📊 Score: {result.overall_score:.2f}")
            print(f"🎯 Recommendation: {result.recommendation}")
            
            for criterion, eval_result in result.evaluations.items():
                print(f"  {criterion}: {eval_result.answer} ({eval_result.confidence:.2f})")
        else:
            print("❌ Analysis failed")
    else:
        print("❌ No documents found")


if __name__ == "__main__":
    main()
