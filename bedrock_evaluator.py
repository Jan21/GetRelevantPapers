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

from markdown_parser import DocumentDatabase
from vector_store import SimpleVectorStore, CriteriaAnalyzer
from analyzer import CriterionResult, PaperResult


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
        
        # BUILD ONE MEGA PROMPT FOR ALL CRITERIA
        mega_prompt = f"""Analyze this research paper and evaluate it against ALL 5 criteria below.
        
Paper Title: {title}

Evaluate each criterion and respond with a JSON object containing all 5 evaluations.

CRITERIA TO EVALUATE:

1. PyTorch Framework - Does it use PyTorch?
Relevant content:
{all_content['pytorch'][:800]}

2. Supervised Learning - Does it focus on supervised learning?
Relevant content:
{all_content['supervised'][:800]}

3. Small Dataset - Works with ≤100K samples (CIFAR, MNIST)?
Relevant content:
{all_content['small_dataset'][:800]}

4. Quick Training - Can train in ≤24 hours on single GPU?
Relevant content:
{all_content['quick_training'][:800]}

5. Public Repository - Has public code repo available NOW?
Relevant content:
{all_content['has_repo'][:800]}

Respond with ONLY this JSON format (no other text):
{{
    "pytorch": {{"answer": "Yes|No|Unknown", "confidence": 0.0-1.0, "evidence": "brief finding"}},
    "supervised": {{"answer": "Yes|No|Unknown", "confidence": 0.0-1.0, "evidence": "brief finding"}},
    "small_dataset": {{"answer": "Yes|No|Unknown", "confidence": 0.0-1.0, "evidence": "brief finding"}},
    "quick_training": {{"answer": "Yes|No|Unknown", "confidence": 0.0-1.0, "evidence": "brief finding"}},
    "has_repo": {{"answer": "Yes|No|Unknown", "confidence": 0.0-1.0, "evidence": "brief finding"}}
}}"""
        
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
    
    def analyze_all_papers_parallel(self, update_callback=None) -> List[PaperResult]:
        """Analyze all papers in parallel"""
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

