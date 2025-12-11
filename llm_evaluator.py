"""
LLM-based Paper Evaluator using OpenRouter Free Models and LangChain

Uses free LLM models to evaluate papers against criteria instead of regex patterns.
More sophisticated analysis that can understand context and nuance.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Install with: pip install langchain")

import aiohttp
import requests

from markdown_parser import DocumentDatabase
from vector_store import SimpleVectorStore, CriteriaAnalyzer
from analyzer import CriterionResult, PaperResult


@dataclass
class LLMEvaluationResult:
    """Result from LLM evaluation"""
    criterion_name: str
    answer: str  # "Yes", "No", "Unknown"
    confidence: float
    evidence: str
    reasoning: str
    raw_response: str


class OpenRouterClient:
    """Client for OpenRouter free models"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Free models available on OpenRouter
        self.free_models = [
            "meta-llama/llama-3.2-3b-instruct:free",
            "meta-llama/llama-3.2-1b-instruct:free", 
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-2-9b-it:free"
        ]
        
        # Default to fastest free model
        self.model = "meta-llama/llama-3.2-3b-instruct:free"
        
    async def chat_completion(self, messages: List[Dict], model: Optional[str] = None) -> str:
        """Make async chat completion request to OpenRouter"""
        model = model or self.model
        
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:3000",  # Required for free tier
            "X-Title": "Deep Researcher Markdown Analysis"  # Required for free tier
        }
        
        # Add API key if available (not required for free models)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,  # Low temperature for consistent analysis
            "max_tokens": 1000,
            "top_p": 0.9
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API error {response.status}: {error_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    def sync_chat_completion(self, messages: List[Dict], model: Optional[str] = None) -> str:
        """Synchronous version for compatibility"""
        model = model or self.model
        
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:3000",  # Required for free tier
            "X-Title": "Deep Researcher Markdown Analysis"  # Required for free tier
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000,
            "top_p": 0.9
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error {response.status_code}: {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


class LLMPaperEvaluator:
    """LLM-based paper evaluator using OpenRouter and LangChain"""
    
    def __init__(self, 
                 document_db: DocumentDatabase,
                 vector_store: SimpleVectorStore,
                 api_key: Optional[str] = None):
        self.document_db = document_db
        self.vector_store = vector_store
        self.criteria_analyzer = CriteriaAnalyzer(vector_store)
        self.openrouter = OpenRouterClient(api_key)
        
        # Criteria definitions with LLM prompts
        self.criteria = {
            'pytorch': {
                'name': 'PyTorch Framework',
                'description': 'The paper uses PyTorch framework for implementation',
                'required': True,
                'weight': 1.0,
                'system_prompt': """You are an expert at identifying deep learning frameworks in research papers. 
Your task is to determine if a paper uses PyTorch as its primary implementation framework.""",
                'evaluation_prompt': """Analyze the following paper content to determine if it uses PyTorch framework.

Look for:
1. Explicit mentions of "PyTorch", "torch", "torchvision"
2. Code snippets showing PyTorch imports (import torch, torch.nn, etc.)
3. References to PyTorch-specific features or optimizations
4. GitHub repositories indicating PyTorch usage

Paper content:
{content}

Respond in this exact JSON format:
{{
    "answer": "Yes|No|Unknown",
    "confidence": 0.0-1.0,
    "evidence": "Direct quotes or specific indicators from the paper",
    "reasoning": "Brief explanation of your decision"
}}"""
            },
            
            'supervised': {
                'name': 'Supervised Learning',
                'description': 'The paper focuses on supervised learning methods',
                'required': True,
                'weight': 1.0,
                'system_prompt': """You are an expert at identifying machine learning paradigms in research papers.
Your task is to determine if a paper primarily uses supervised learning methods.""",
                'evaluation_prompt': """Analyze the following paper content to determine if it focuses on supervised learning.

Supervised learning indicators:
1. Uses labeled datasets with ground truth
2. Training with input-output pairs
3. Loss functions like cross-entropy, MSE with labels
4. Evaluation metrics like accuracy, precision, recall
5. Mentions of "supervised learning", "classification", "regression"

NOT supervised learning:
- Self-supervised learning (contrastive, masked modeling)
- Unsupervised learning (clustering, autoencoders)
- Reinforcement learning
- Semi-supervised (unless supervised is primary focus)

Paper content:
{content}

Respond in this exact JSON format:
{{
    "answer": "Yes|No|Unknown",
    "confidence": 0.0-1.0,
    "evidence": "Direct quotes or specific indicators from the paper",
    "reasoning": "Brief explanation of your decision"
}}"""
            },
            
            'small_dataset': {
                'name': 'Small Dataset',
                'description': 'The method works with small datasets (≤100K samples)',
                'required': False,
                'weight': 0.6,
                'system_prompt': """You are an expert at analyzing dataset sizes and data efficiency in research papers.
Your task is to determine if a paper works with small datasets (≤100K samples).""",
                'evaluation_prompt': """Analyze the following paper content to determine if it works with small datasets.

Small datasets (≤100K samples):
- CIFAR-10 (60K), CIFAR-100 (60K)
- MNIST (70K), Fashion-MNIST (60K)
- SVHN (73K)
- Custom datasets with ≤100K samples
- Papers emphasizing data efficiency or few-shot learning

Large datasets (>100K samples):
- ImageNet-1K (1.2M), ImageNet-21K (14M)
- Large-scale web datasets
- Papers requiring massive datasets

Paper content:
{content}

Respond in this exact JSON format:
{{
    "answer": "Yes|No|Unknown",
    "confidence": 0.0-1.0,
    "evidence": "Dataset names, sizes, or data efficiency claims",
    "reasoning": "Brief explanation of your decision"
}}"""
            },
            
            'quick_training': {
                'name': 'Quick Training',
                'description': 'The model can be trained quickly (≤24 hours on single GPU)',
                'required': False,
                'weight': 0.4,
                'system_prompt': """You are an expert at analyzing computational requirements and training efficiency in research papers.
Your task is to determine if a model can be trained quickly (≤24 hours on single GPU).""",
                'evaluation_prompt': """Analyze the following paper content to determine if the model can be trained quickly.

Quick training indicators:
1. Explicit training time ≤24 hours on single GPU
2. Lightweight models (few parameters)
3. Efficient architectures (MobileNet, EfficientNet style)
4. Claims of fast convergence or training efficiency
5. Single GPU training setups

Slow training indicators:
1. Multi-day training requirements
2. Multi-GPU or TPU requirements
3. Very large models (>1B parameters)
4. Distributed training setups

Paper content:
{content}

Respond in this exact JSON format:
{{
    "answer": "Yes|No|Unknown",
    "confidence": 0.0-1.0,
    "evidence": "Training time, hardware requirements, or efficiency claims",
    "reasoning": "Brief explanation of your decision"
}}"""
            },
            
            'has_repo': {
                'name': 'Public Repository',
                'description': 'The paper provides a public code repository',
                'required': True,
                'weight': 1.0,
                'system_prompt': """You are an expert at identifying code availability in research papers.
Your task is to determine if a paper provides a public code repository.""",
                'evaluation_prompt': """Analyze the following paper content to determine if it provides a public code repository.

Public repository indicators:
1. GitHub, GitLab, or similar URLs
2. "Code available at [URL]"
3. "Implementation available"
4. Open source mentions
5. Repository links in text

NOT public repository:
1. "Code available upon request"
2. "Will be released upon acceptance"
3. "Available upon publication" (future promise)
4. No code availability mentioned

Paper content:
{content}

Respond in this exact JSON format:
{{
    "answer": "Yes|No|Unknown",
    "confidence": 0.0-1.0,
    "evidence": "URLs, availability statements, or repository mentions",
    "reasoning": "Brief explanation of your decision"
}}"""
            }
        }
    
    async def evaluate_criterion_async(self, 
                                     doc_id: str, 
                                     criterion_name: str, 
                                     content: str) -> LLMEvaluationResult:
        """Evaluate a single criterion using LLM"""
        criterion_def = self.criteria[criterion_name]
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": criterion_def['system_prompt']},
            {"role": "user", "content": criterion_def['evaluation_prompt'].format(content=content)}
        ]
        
        try:
            # Get LLM response
            response = await self.openrouter.chat_completion(messages)
            
            # Parse JSON response
            result = self._parse_llm_response(response, criterion_name)
            result.raw_response = response
            
            return result
            
        except Exception as e:
            print(f"LLM evaluation failed for {criterion_name}: {e}")
            return LLMEvaluationResult(
                criterion_name=criterion_name,
                answer="Unknown",
                confidence=0.0,
                evidence=f"LLM evaluation failed: {str(e)}",
                reasoning="Technical error during evaluation",
                raw_response=""
            )
    
    def evaluate_criterion_sync(self, 
                               doc_id: str, 
                               criterion_name: str, 
                               content: str) -> LLMEvaluationResult:
        """Synchronous version of criterion evaluation"""
        criterion_def = self.criteria[criterion_name]
        
        messages = [
            {"role": "system", "content": criterion_def['system_prompt']},
            {"role": "user", "content": criterion_def['evaluation_prompt'].format(content=content)}
        ]
        
        try:
            response = self.openrouter.sync_chat_completion(messages)
            result = self._parse_llm_response(response, criterion_name)
            result.raw_response = response
            return result
            
        except Exception as e:
            print(f"LLM evaluation failed for {criterion_name}: {e}")
            return LLMEvaluationResult(
                criterion_name=criterion_name,
                answer="Unknown",
                confidence=0.0,
                evidence=f"LLM evaluation failed: {str(e)}",
                reasoning="Technical error during evaluation",
                raw_response=""
            )
    
    def _parse_llm_response(self, response: str, criterion_name: str) -> LLMEvaluationResult:
        """Parse LLM JSON response with fallback"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                return LLMEvaluationResult(
                    criterion_name=criterion_name,
                    answer=data.get("answer", "Unknown"),
                    confidence=float(data.get("confidence", 0.5)),
                    evidence=data.get("evidence", ""),
                    reasoning=data.get("reasoning", ""),
                    raw_response=response
                )
            else:
                # Fallback parsing
                return self._fallback_parse(response, criterion_name)
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing failed for {criterion_name}: {e}")
            return self._fallback_parse(response, criterion_name)
    
    def _fallback_parse(self, response: str, criterion_name: str) -> LLMEvaluationResult:
        """Fallback parsing when JSON fails"""
        response_lower = response.lower()
        
        # Simple keyword detection
        if any(word in response_lower for word in ["yes", "true", "positive", "meets"]):
            answer = "Yes"
            confidence = 0.6
        elif any(word in response_lower for word in ["no", "false", "negative", "does not"]):
            answer = "No"
            confidence = 0.6
        else:
            answer = "Unknown"
            confidence = 0.3
        
        return LLMEvaluationResult(
            criterion_name=criterion_name,
            answer=answer,
            confidence=confidence,
            evidence="Parsed from free-form response",
            reasoning="Fallback parsing used",
            raw_response=response
        )
    
    def analyze_paper(self, doc_id: str, use_async: bool = False) -> Optional[PaperResult]:
        """Analyze a paper using LLM evaluation"""
        # Get document metadata and sections
        doc_meta = self.document_db.get_document(doc_id)
        if not doc_meta:
            print(f"Document {doc_id} not found")
            return None
        
        sections = self.document_db.get_document_sections(doc_id)
        if not sections:
            print(f"No sections found for document {doc_id}")
            return None
        
        print(f"Analyzing with LLM: {doc_meta['title']}")
        
        # Analyze each criterion
        evaluations = {}
        
        for criterion_name in self.criteria.keys():
            # Get relevant sections using vector store
            relevant_sections = self.criteria_analyzer.get_section_content_for_criteria(
                doc_id, criterion_name, top_k=3
            )
            
            # Collect relevant content
            content_parts = []
            section_keys = []
            
            for section_info in relevant_sections:
                key = section_info['section_key']
                if key in sections:
                    content_parts.append(f"## {key.replace('_', ' ').title()}\n{sections[key]}")
                    section_keys.append(key)
            
            # If no relevant sections, use all content (truncated)
            if not content_parts:
                all_content = "\n\n".join([f"## {k.replace('_', ' ').title()}\n{v}" for k, v in sections.items()])
                content = all_content[:4000] + "..." if len(all_content) > 4000 else all_content
                section_keys = list(sections.keys())
            else:
                content = "\n\n".join(content_parts)
            
            # Evaluate criterion with LLM
            if use_async:
                # For async, you'd need to handle this differently
                llm_result = self.evaluate_criterion_sync(doc_id, criterion_name, content)
            else:
                llm_result = self.evaluate_criterion_sync(doc_id, criterion_name, content)
            
            # Convert to CriterionResult format
            evaluations[criterion_name] = CriterionResult(
                name=criterion_name,
                answer=llm_result.answer,
                confidence=llm_result.confidence,
                evidence=llm_result.evidence,
                sections=section_keys
            )
        
        # Calculate overall score and recommendation
        overall_score = self._calculate_score(evaluations)
        recommendation = self._make_recommendation(evaluations, overall_score)
        summary = f"{doc_meta['title']} - {recommendation} (LLM Score: {overall_score:.2f})"
        
        return PaperResult(
            doc_id=doc_id,
            title=doc_meta['title'],
            evaluations=evaluations,
            overall_score=overall_score,
            recommendation=recommendation,
            summary=summary
        )
    
    def _calculate_score(self, evaluations: Dict[str, CriterionResult]) -> float:
        """Calculate weighted overall score"""
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
            else:  # Unknown
                score = 0.5 * (1 - evaluation.confidence)
            
            weighted_score += score * weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _make_recommendation(self, 
                           evaluations: Dict[str, CriterionResult], 
                           overall_score: float) -> str:
        """Make recommendation based on evaluations"""
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
    
    def analyze_all_papers(self) -> List[PaperResult]:
        """Analyze all papers using LLM"""
        documents = self.document_db.list_documents()
        results = []
        
        for doc_info in documents:
            result = self.analyze_paper(doc_info['doc_id'])
            if result:
                results.append(result)
        
        return results
    
    def compare_with_regex(self, doc_id: str) -> Dict:
        """Compare LLM results with regex-based results"""
        from analyzer import PaperAnalyzer
        
        # Get LLM results
        llm_result = self.analyze_paper(doc_id)
        
        # Get regex results
        regex_analyzer = PaperAnalyzer(self.document_db, self.vector_store)
        regex_result = regex_analyzer.analyze_paper(doc_id)
        
        if not llm_result or not regex_result:
            return {}
        
        comparison = {
            'paper_title': llm_result.title,
            'llm_score': llm_result.overall_score,
            'regex_score': regex_result.overall_score,
            'llm_recommendation': llm_result.recommendation,
            'regex_recommendation': regex_result.recommendation,
            'criteria_comparison': {}
        }
        
        for criterion_name in self.criteria.keys():
            llm_eval = llm_result.evaluations[criterion_name]
            regex_eval = regex_result.evaluations[criterion_name]
            
            comparison['criteria_comparison'][criterion_name] = {
                'llm_answer': llm_eval.answer,
                'regex_answer': regex_eval.answer,
                'llm_confidence': llm_eval.confidence,
                'regex_confidence': regex_eval.confidence,
                'agreement': llm_eval.answer == regex_eval.answer
            }
        
        return comparison
    
    def save_results(self, results: List[PaperResult], output_file: Optional[Path] = None):
        """Save results to JSON (compatible with PaperAnalyzer)"""
        if output_file is None:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"llm_analysis_{timestamp}.json")
        
        data = {
            'results': [r.to_dict() for r in results],
            'summary': {
                'total': len(results),
                'include': sum(1 for r in results if r.recommendation == "Include"),
                'exclude': sum(1 for r in results if r.recommendation == "Exclude"),
                'review': sum(1 for r in results if r.recommendation == "Review"),
                'method': 'LLM'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 LLM results saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Initialize components
    db = DocumentDatabase(Path("markdown_db"))
    vector_store = SimpleVectorStore(Path("vector_store"))
    
    # Initialize LLM evaluator
    llm_evaluator = LLMPaperEvaluator(db, vector_store)
    
    print("LLM Paper Evaluator ready!")
    print(f"Using model: {llm_evaluator.openrouter.model}")
    print(f"Available criteria: {list(llm_evaluator.criteria.keys())}")
