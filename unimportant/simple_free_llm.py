#!/usr/bin/env python3
"""
Simple FREE LLM Evaluator - No API keys, no huge downloads
Uses Hugging Face Inference API (free tier) or falls back to enhanced regex
"""

import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time

from markdown_parser import DocumentDatabase
from vector_store import SimpleVectorStore, CriteriaAnalyzer
from analyzer import CriterionResult, PaperResult


@dataclass
class SimpleLLMResult:
    """Result from simple LLM evaluation"""
    criterion_name: str
    answer: str  # "Yes", "No", "Unknown"
    confidence: float
    evidence: str
    reasoning: str


class SimpleFreeEvaluator:
    """Simple free LLM evaluator using Hugging Face Inference API (no auth required)"""
    
    def __init__(self, document_db: DocumentDatabase, vector_store: SimpleVectorStore):
        self.document_db = document_db
        self.vector_store = vector_store
        self.criteria_analyzer = CriteriaAnalyzer(vector_store)
        
        # Hugging Face Inference API endpoint (free, no auth required for some models)
        self.hf_api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        
        # Define criteria with simple prompts
        self.criteria = {
            'pytorch': {
                'description': 'Uses PyTorch framework',
                'keywords_positive': ['pytorch', 'torch.', 'torchvision', 'torch nn', 'torch optim'],
                'keywords_negative': ['tensorflow', 'keras', 'jax', 'tf.', 'from tensorflow'],
                'prompt': 'Does this text mention PyTorch framework? Answer yes or no:'
            },
            'supervised': {
                'description': 'Uses supervised learning',
                'keywords_positive': ['supervised', 'classification', 'labeled data', 'ground truth', 'cross-entropy', 'training labels'],
                'keywords_negative': ['unsupervised', 'self-supervised', 'reinforcement learning', 'semi-supervised'],
                'prompt': 'Does this text describe supervised learning? Answer yes or no:'
            },
            'small_dataset': {
                'description': 'Uses small datasets',
                'keywords_positive': ['cifar', 'mnist', 'small-scale', 'small dataset', 'limited data', 'few-shot'],
                'keywords_negative': ['imagenet', 'large-scale', 'million images', 'billion parameters', 'imagenet-1k'],
                'prompt': 'Does this text use small datasets like CIFAR or MNIST? Answer yes or no:'
            },
            'quick_training': {
                'description': 'Has quick training time',
                'keywords_positive': ['single gpu', 'hours', 'lightweight', 'fast', 'efficient training', 'quick', 'minutes'],
                'keywords_negative': ['multi-day', 'weeks', 'months', 'distributed', 'large models', 'multi-gpu'],
                'prompt': 'Does this text describe quick or efficient training? Answer yes or no:'
            },
            'has_repo': {
                'description': 'Has code repository',
                'keywords_positive': ['github.com', 'gitlab.com', 'code available', 'implementation available', 'repository', 'open source'],
                'keywords_negative': ['will be released', 'upon request', 'not available'],
                'prompt': 'Does this text mention available code or repository? Answer yes or no:'
            }
        }
    
    def query_huggingface(self, text: str, max_retries: int = 2) -> Optional[str]:
        """Query Hugging Face Inference API (free tier)"""
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "inputs": text,
                "parameters": {
                    "max_length": len(text) + 50,
                    "temperature": 0.1,
                    "do_sample": True
                }
            }
            
            for attempt in range(max_retries):
                response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    print(f"🔄 Model loading, waiting {attempt + 1}s...")
                    time.sleep(attempt + 1)
                else:
                    print(f"⚠️ HF API error {response.status_code}: {response.text}")
                    break
            
            return None
            
        except Exception as e:
            print(f"❌ HF API request failed: {e}")
            return None
    
    def evaluate_criterion_simple(self, doc_id: str, criterion_name: str) -> SimpleLLMResult:
        """Evaluate a single criterion using simple methods"""
        try:
            # Get relevant sections
            relevant_sections = self.criteria_analyzer.get_section_content_for_criteria(
                doc_id, criterion_name, top_k=3
            )
            
            if not relevant_sections:
                return SimpleLLMResult(
                    criterion_name=criterion_name,
                    answer="Unknown",
                    confidence=0.0,
                    evidence="No relevant sections found",
                    reasoning="No content to analyze"
                )
            
            # Combine relevant content
            content = " ".join([section['content_preview'][:150] for section in relevant_sections])
            
            # Try HF API first, fallback to keywords
            hf_result = self.try_huggingface_analysis(criterion_name, content)
            if hf_result:
                return hf_result
            
            # Fallback to enhanced keyword analysis
            return self.evaluate_with_enhanced_keywords(criterion_name, content)
                
        except Exception as e:
            print(f"❌ Error evaluating {criterion_name}: {e}")
            return SimpleLLMResult(
                criterion_name=criterion_name,
                answer="Unknown",
                confidence=0.0,
                evidence=f"Evaluation failed: {str(e)}",
                reasoning="Error during analysis"
            )
    
    def try_huggingface_analysis(self, criterion_name: str, content: str) -> Optional[SimpleLLMResult]:
        """Try to use Hugging Face API for analysis"""
        try:
            criterion = self.criteria[criterion_name]
            
            # Create a simple prompt
            prompt = f"{criterion['prompt']} {content[:200]}"
            
            # Query HF API
            response = self.query_huggingface(prompt)
            
            if response:
                # Extract the generated part
                generated = response[len(prompt):].strip().lower()
                
                # Parse response
                if 'yes' in generated[:20] or 'true' in generated[:20]:
                    answer = "Yes"
                    confidence = 0.7
                    reasoning = f"HF API: {generated[:50]}"
                elif 'no' in generated[:20] or 'false' in generated[:20]:
                    answer = "No"
                    confidence = 0.7
                    reasoning = f"HF API: {generated[:50]}"
                else:
                    return None  # Unclear response, fallback to keywords
                
                return SimpleLLMResult(
                    criterion_name=criterion_name,
                    answer=answer,
                    confidence=confidence,
                    evidence=content[:200],
                    reasoning=reasoning
                )
            
            return None
            
        except Exception as e:
            print(f"⚠️ HF API analysis failed: {e}")
            return None
    
    def evaluate_with_enhanced_keywords(self, criterion_name: str, content: str) -> SimpleLLMResult:
        """Enhanced keyword-based evaluation with better scoring"""
        content_lower = content.lower()
        criterion = self.criteria[criterion_name]
        
        # Count keyword matches
        positive_matches = []
        negative_matches = []
        
        for keyword in criterion['keywords_positive']:
            if keyword in content_lower:
                positive_matches.append(keyword)
        
        for keyword in criterion['keywords_negative']:
            if keyword in content_lower:
                negative_matches.append(keyword)
        
        positive_count = len(positive_matches)
        negative_count = len(negative_matches)
        
        # Enhanced scoring logic
        if positive_count > 0 and negative_count == 0:
            answer = "Yes"
            confidence = min(0.9, 0.6 + (positive_count * 0.1))
            evidence = f"Found: {', '.join(positive_matches[:3])}"
        elif negative_count > 0 and positive_count == 0:
            answer = "No"
            confidence = min(0.9, 0.6 + (negative_count * 0.1))
            evidence = f"Found negative: {', '.join(negative_matches[:3])}"
        elif positive_count > negative_count:
            answer = "Yes"
            confidence = 0.6
            evidence = f"More positive ({positive_count}) than negative ({negative_count})"
        elif negative_count > positive_count:
            answer = "No"
            confidence = 0.6
            evidence = f"More negative ({negative_count}) than positive ({positive_count})"
        else:
            answer = "Unknown"
            confidence = 0.3
            evidence = "No clear evidence"
        
        return SimpleLLMResult(
            criterion_name=criterion_name,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            reasoning="Enhanced keyword analysis"
        )
    
    def analyze_paper(self, doc_id: str) -> Optional[PaperResult]:
        """Analyze a paper using simple free methods"""
        # Get document metadata
        doc_meta = self.document_db.get_document(doc_id)
        if not doc_meta:
            print(f"Document {doc_id} not found")
            return None
        
        print(f"🆓 Analyzing with Free LLM: {doc_meta['title']}")
        
        # Analyze each criterion
        evaluations = {}
        for criterion_name in self.criteria.keys():
            print(f"  📊 Evaluating: {criterion_name}")
            
            result = self.evaluate_criterion_simple(doc_id, criterion_name)
            
            # Convert to CriterionResult
            evaluations[criterion_name] = CriterionResult(
                name=criterion_name,
                answer=result.answer,
                confidence=result.confidence,
                evidence=result.evidence,
                sections=[]
            )
        
        # Calculate overall score (same logic as regex analyzer)
        total_score = 0.0
        total_weight = 0.0
        
        # Weights for each criterion
        weights = {
            'pytorch': 1.0,      # Required
            'supervised': 1.0,   # Required
            'small_dataset': 0.6,
            'quick_training': 0.4,
            'has_repo': 1.0     # Required
        }
        
        for criterion_name, evaluation in evaluations.items():
            weight = weights.get(criterion_name, 1.0)
            
            if evaluation.answer == "Yes":
                score = evaluation.confidence * weight
            elif evaluation.answer == "No":
                # Required criteria get 0, optional get penalty
                if weight >= 1.0:  # Required criteria
                    score = 0.0
                else:
                    score = 0.2 * weight  # Small penalty for optional
            else:  # Unknown
                score = 0.5 * weight
            
            total_score += score
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.5
        
        # Determine recommendation
        if overall_score >= 0.8:
            recommendation = "Include"
        elif overall_score >= 0.5:
            recommendation = "Review"
        else:
            recommendation = "Exclude"
        
        summary = f"{doc_meta['title']} - {recommendation} (Free LLM Score: {overall_score:.2f})"
        
        return PaperResult(
            doc_id=doc_id,
            title=doc_meta['title'],
            evaluations=evaluations,
            overall_score=overall_score,
            recommendation=recommendation,
            summary=summary
        )
    
    def save_results(self, results: List[PaperResult], output_file: Optional[Path] = None):
        """Save results to JSON"""
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
                'method': 'Free LLM'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Free LLM results saved to: {output_file}")


def main():
    """Test the simple free LLM evaluator"""
    # Setup
    db_path = Path("markdown_db")
    vector_path = Path("vector_store")
    
    db = DocumentDatabase(db_path)
    vector_store = SimpleVectorStore(vector_path)
    evaluator = SimpleFreeEvaluator(db, vector_store)
    
    # Test on first document
    docs = db.list_documents()
    if docs:
        doc_info = docs[0]
        result = evaluator.analyze_paper(doc_info['doc_id'])
        if result:
            print(f"\n✅ Analysis complete!")
            print(f"📊 Score: {result.overall_score:.2f}")
            print(f"🎯 Recommendation: {result.recommendation}")
            
            for criterion, eval_result in result.evaluations.items():
                print(f"  {criterion}: {eval_result.answer} ({eval_result.confidence:.1f})")


if __name__ == "__main__":
    main()
