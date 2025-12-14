"""
Paper Analysis System - Compact Version

Analyzes markdown papers against Deep Researcher criteria.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.core.markdown_parser import MarkdownParser, DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer


@dataclass
class CriterionResult:
    """Result for a single criterion"""
    name: str
    answer: str  # "Yes", "No", "Unknown"
    confidence: float
    evidence: str
    sections: List[str]


@dataclass
class PaperResult:
    """Complete paper analysis result"""
    doc_id: str
    title: str
    evaluations: Dict[str, CriterionResult]
    overall_score: float
    recommendation: str
    summary: str
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'overall_score': self.overall_score,
            'recommendation': self.recommendation,
            'summary': self.summary,
            'evaluations': {
                name: {
                    'answer': eval_result.answer,
                    'confidence': eval_result.confidence,
                    'evidence': eval_result.evidence
                }
                for name, eval_result in self.evaluations.items()
            }
        }


class PaperAnalyzer:
    """Analyze papers against criteria"""
    
    def __init__(self, document_db: DocumentDatabase, vector_store: SimpleVectorStore):
        self.document_db = document_db
        self.vector_store = vector_store
        self.criteria_analyzer = CriteriaAnalyzer(vector_store)
        
        # Simplified criteria patterns
        self.criteria = {
            'pytorch': {
                'patterns': [r'pytorch', r'torch\.', r'torchvision'],
                'negative': [r'tensorflow', r'keras', r'jax'],
                'required': True, 'weight': 1.0
            },
            'supervised': {
                'patterns': [r'supervised', r'classification', r'labeled.*data', r'cross-entropy'],
                'negative': [r'unsupervised', r'self-supervised', r'reinforcement'],
                'required': True, 'weight': 1.0
            },
            'small_dataset': {
                'patterns': [r'cifar', r'mnist', r'small.*dataset', r'60,?000.*samples'],
                'negative': [r'imagenet', r'million.*images', r'large.*scale'],
                'required': False, 'weight': 0.6
            },
            'quick_training': {
                'patterns': [r'\d+\s+hours.*training', r'single.*gpu', r'fast.*training'],
                'negative': [r'days.*training', r'multi.*gpu', r'distributed'],
                'required': False, 'weight': 0.4
            },
            'has_repo': {
                'patterns': [r'github\.com', r'code.*available', r'repository'],
                'negative': [r'upon.*request', r'will.*be.*released'],
                'required': True, 'weight': 1.0
            }
        }
    
    def analyze_paper(self, doc_id: str) -> Optional[PaperResult]:
        """Analyze single paper"""
        doc_meta = self.document_db.get_document(doc_id)
        sections = self.document_db.get_document_sections(doc_id)
        
        if not doc_meta or not sections:
            return None
        
        evaluations = {}
        for criterion_name, criterion_def in self.criteria.items():
            # Get relevant sections
            relevant_sections = self.criteria_analyzer.get_section_content_for_criteria(
                doc_id, criterion_name, top_k=3
            )
            
            # Collect text
            text = ""
            section_keys = []
            for section_info in relevant_sections:
                key = section_info['section_key']
                if key in sections:
                    text += f"\n{sections[key]}"
                    section_keys.append(key)
            
            if not text:
                text = "\n".join(sections.values())
                section_keys = list(sections.keys())
            
            # Evaluate criterion
            result = self._evaluate_criterion(text, criterion_def, criterion_name)
            result.sections = section_keys
            evaluations[criterion_name] = result
        
        # Calculate overall score
        overall_score = self._calculate_score(evaluations)
        recommendation = self._make_recommendation(evaluations, overall_score)
        summary = f"{doc_meta['title']} - {recommendation} (Score: {overall_score:.2f})"
        
        return PaperResult(
            doc_id=doc_id,
            title=doc_meta['title'],
            evaluations=evaluations,
            overall_score=overall_score,
            recommendation=recommendation,
            summary=summary
        )
    
    def _evaluate_criterion(self, text: str, criterion_def: Dict, name: str) -> CriterionResult:
        """Evaluate text against criterion"""
        text_lower = text.lower()
        
        # Count positive matches
        positive_count = 0
        positive_matches = []
        for pattern in criterion_def['patterns']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            positive_count += len(matches)
            positive_matches.extend(matches[:2])
        
        # Count negative matches
        negative_count = 0
        negative_matches = []
        for pattern in criterion_def.get('negative', []):
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            negative_count += len(matches)
            negative_matches.extend(matches[:2])
        
        # Determine result
        if positive_count > 0 and negative_count == 0:
            answer = "Yes"
            confidence = min(0.9, 0.6 + positive_count * 0.1)
            evidence = f"Found: {', '.join(positive_matches[:3])}"
        elif negative_count > 0 and positive_count == 0:
            answer = "No"
            confidence = min(0.9, 0.6 + negative_count * 0.1)
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
        
        return CriterionResult(
            name=name,
            answer=answer,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=evidence,
            sections=[]
        )
    
    def _calculate_score(self, evaluations: Dict[str, CriterionResult]) -> float:
        """Calculate weighted score"""
        total_weight = 0.0
        weighted_score = 0.0
        
        for criterion_name, evaluation in evaluations.items():
            weight = self.criteria[criterion_name]['weight']
            total_weight += weight
            
            if evaluation.answer == "Yes":
                score = 1.0 * evaluation.confidence
            elif evaluation.answer == "No":
                score = 0.0
            else:
                score = 0.5 * (1 - evaluation.confidence)
            
            weighted_score += score * weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _make_recommendation(self, evaluations: Dict[str, CriterionResult], score: float) -> str:
        """Make recommendation"""
        # Check required criteria
        for criterion_name, evaluation in evaluations.items():
            if self.criteria[criterion_name]['required']:
                if evaluation.answer == "No" and evaluation.confidence > 0.7:
                    return "Exclude"
        
        if score >= 0.8:
            return "Include"
        elif score >= 0.5:
            return "Review"
        else:
            return "Exclude"
    
    def analyze_all(self) -> List[PaperResult]:
        """Analyze all papers"""
        documents = self.document_db.list_documents()
        results = []
        
        for doc_info in documents:
            print(f"Analyzing: {doc_info['title']}")
            result = self.analyze_paper(doc_info['doc_id'])
            if result:
                results.append(result)
        
        return results
    
    def save_results(self, results: List[PaperResult], output_file: Optional[Path] = None):
        """Save results to JSON"""
        if output_file is None:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"regex_analysis_{timestamp}.json")
        
        data = {
            'results': [r.to_dict() for r in results],
            'summary': {
                'total': len(results),
                'include': sum(1 for r in results if r.recommendation == "Include"),
                'exclude': sum(1 for r in results if r.recommendation == "Exclude"),
                'review': sum(1 for r in results if r.recommendation == "Review"),
                'method': 'Regex'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        
        print(f"Results saved to {output_file}")
