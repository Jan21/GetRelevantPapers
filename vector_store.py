"""
Simple Vector Store for Section Headings

Creates embeddings for section headings and enables semantic search
to find the most relevant sections for analysis criteria.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
import re


@dataclass
class SectionEmbedding:
    """Embedding for a document section"""
    doc_id: str
    document_title: str
    section_key: str
    section_heading: str
    embedding: np.ndarray
    content_preview: str  # First 200 chars of content


class SimpleVectorStore:
    """Simple vector store using TF-IDF for semantic search of section headings"""
    
    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = self.store_path / "embeddings.pkl"
        self.metadata_file = self.store_path / "metadata.json"
        
        # Simple TF-IDF vocabulary
        self.vocabulary = {}
        self.idf_scores = {}
        self.embeddings: List[SectionEmbedding] = []
        
        # Load existing data
        self._load_store()
    
    def _load_store(self):
        """Load existing embeddings and metadata"""
        if self.embeddings_file.exists() and self.metadata_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.vocabulary = data.get('vocabulary', {})
                    self.idf_scores = data.get('idf_scores', {})
                
                print(f"Loaded {len(self.embeddings)} embeddings from store")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.embeddings = []
                self.vocabulary = {}
                self.idf_scores = {}
    
    def _save_store(self):
        """Save embeddings and metadata to disk"""
        data = {
            'embeddings': self.embeddings,
            'vocabulary': self.vocabulary,
            'idf_scores': self.idf_scores
        }
        
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Save readable metadata
        metadata = {
            'embedding_count': len(self.embeddings),
            'vocabulary_size': len(self.vocabulary),
            'updated_at': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase, remove special chars, split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [token for token in tokens if len(token) > 2]  # Filter short tokens
    
    def _build_vocabulary(self, documents: List[str]):
        """Build vocabulary from all documents"""
        all_tokens = set()
        
        for doc in documents:
            tokens = self._tokenize(doc)
            all_tokens.update(tokens)
        
        # Create vocabulary mapping
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        
        # Calculate IDF scores
        self._calculate_idf(documents)
    
    def _calculate_idf(self, documents: List[str]):
        """Calculate IDF scores for vocabulary"""
        doc_count = len(documents)
        token_doc_counts = {}
        
        # Count documents containing each token
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                token_doc_counts[token] = token_doc_counts.get(token, 0) + 1
        
        # Calculate IDF scores
        self.idf_scores = {}
        for token, vocab_idx in self.vocabulary.items():
            doc_freq = token_doc_counts.get(token, 1)
            idf = np.log(doc_count / doc_freq)
            self.idf_scores[token] = idf
    
    def _create_tfidf_vector(self, text: str) -> np.ndarray:
        """Create TF-IDF vector for text"""
        tokens = self._tokenize(text)
        vector = np.zeros(len(self.vocabulary))
        
        if not tokens:
            return vector
        
        # Calculate term frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Create TF-IDF vector
        for token, count in token_counts.items():
            if token in self.vocabulary:
                vocab_idx = self.vocabulary[token]
                tf = count / len(tokens)  # Term frequency
                idf = self.idf_scores.get(token, 1.0)  # Inverse document frequency
                vector[vocab_idx] = tf * idf
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def add_sections(self, sections_data: List[Dict]):
        """Add sections to the vector store"""
        # Prepare documents for vocabulary building
        documents = []
        new_embeddings = []
        
        for section in sections_data:
            # Combine heading and content for embedding
            text = f"{section['section_heading']} {section['content'][:500]}"
            documents.append(text)
        
        # Build vocabulary if empty or rebuild with new data
        if not self.vocabulary:
            all_docs = documents
            # Add existing documents if any
            for emb in self.embeddings:
                existing_text = f"{emb.section_heading} {emb.content_preview}"
                all_docs.append(existing_text)
            
            self._build_vocabulary(all_docs)
        
        # Create embeddings for new sections
        for section in sections_data:
            text = f"{section['section_heading']} {section['content'][:500]}"
            embedding_vector = self._create_tfidf_vector(text)
            
            section_embedding = SectionEmbedding(
                doc_id=section['doc_id'],
                document_title=section['document_title'],
                section_key=section['section_key'],
                section_heading=section['section_heading'],
                embedding=embedding_vector,
                content_preview=section['content'][:200]
            )
            
            new_embeddings.append(section_embedding)
        
        self.embeddings.extend(new_embeddings)
        self._save_store()
        
        print(f"Added {len(new_embeddings)} section embeddings")
    
    def search_similar_sections(self, query: str, top_k: int = 10) -> List[Tuple[SectionEmbedding, float]]:
        """Search for sections similar to query"""
        if not self.embeddings:
            return []
        
        # Create query vector
        query_vector = self._create_tfidf_vector(query)
        
        # Calculate similarities
        similarities = []
        for embedding in self.embeddings:
            # Cosine similarity
            similarity = np.dot(query_vector, embedding.embedding)
            similarities.append((embedding, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_section_clusters(self, threshold: float = 0.3) -> Dict[str, List[SectionEmbedding]]:
        """Group similar sections together"""
        if not self.embeddings:
            return {}
        
        clusters = {}
        processed = set()
        
        for i, embedding in enumerate(self.embeddings):
            if i in processed:
                continue
            
            # Start new cluster
            cluster_key = f"cluster_{len(clusters)}"
            cluster = [embedding]
            processed.add(i)
            
            # Find similar embeddings
            for j, other_embedding in enumerate(self.embeddings):
                if j in processed or i == j:
                    continue
                
                # Calculate similarity
                similarity = np.dot(embedding.embedding, other_embedding.embedding)
                
                if similarity > threshold:
                    cluster.append(other_embedding)
                    processed.add(j)
            
            clusters[cluster_key] = cluster
        
        return clusters
    
    def analyze_heading_patterns(self) -> Dict[str, int]:
        """Analyze common heading patterns across documents"""
        heading_counts = {}
        
        for embedding in self.embeddings:
            # Normalize heading for pattern analysis
            normalized = self._normalize_heading(embedding.section_heading)
            heading_counts[normalized] = heading_counts.get(normalized, 0) + 1
        
        # Sort by frequency
        return dict(sorted(heading_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _normalize_heading(self, heading: str) -> str:
        """Normalize heading for pattern analysis"""
        # Remove numbers, convert to lowercase, remove extra spaces
        normalized = re.sub(r'\d+\.?\s*', '', heading.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            'total_embeddings': len(self.embeddings),
            'vocabulary_size': len(self.vocabulary),
            'unique_documents': len(set(emb.doc_id for emb in self.embeddings)),
            'avg_embedding_norm': np.mean([np.linalg.norm(emb.embedding) for emb in self.embeddings]) if self.embeddings else 0
        }


class CriteriaAnalyzer:
    """Analyze which section headings are useful for evaluation criteria"""
    
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store
        
        # Criteria from Deep Researcher project
        self.criteria_keywords = {
            'pytorch': [
                'implementation', 'framework', 'code', 'experiments', 'methodology',
                'model', 'training', 'architecture', 'setup', 'technical'
            ],
            'supervised': [
                'methodology', 'approach', 'model', 'training', 'learning',
                'dataset', 'data', 'experiments', 'evaluation'
            ],
            'small_dataset': [
                'dataset', 'data', 'experiments', 'evaluation', 'results',
                'benchmarks', 'performance', 'testing'
            ],
            'quick_training': [
                'experiments', 'training', 'performance', 'efficiency',
                'computational', 'time', 'resources', 'setup'
            ],
            'has_repo': [
                'code', 'implementation', 'availability', 'repository',
                'github', 'source', 'software', 'appendix'
            ]
        }
    
    def analyze_useful_headings(self) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze which headings are most useful for each criterion"""
        results = {}
        
        for criterion, keywords in self.criteria_keywords.items():
            # Create query from keywords
            query = ' '.join(keywords)
            
            # Search for relevant sections
            similar_sections = self.vector_store.search_similar_sections(query, top_k=20)
            
            # Group by heading pattern
            heading_scores = {}
            for section_emb, score in similar_sections:
                normalized_heading = self.vector_store._normalize_heading(section_emb.section_heading)
                
                if normalized_heading not in heading_scores:
                    heading_scores[normalized_heading] = []
                heading_scores[normalized_heading].append(score)
            
            # Calculate average scores for each heading
            avg_scores = {}
            for heading, scores in heading_scores.items():
                avg_scores[heading] = np.mean(scores)
            
            # Sort by relevance
            sorted_headings = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            results[criterion] = sorted_headings[:10]  # Top 10 headings
        
        return results
    
    def get_section_content_for_criteria(self, doc_id: str, criterion: str, top_k: int = 5) -> List[Dict]:
        """Get the most relevant section content for a specific criterion and document"""
        # Get keywords for criterion
        keywords = self.criteria_keywords.get(criterion, [])
        if not keywords:
            return []
        
        query = ' '.join(keywords)
        
        # Search for relevant sections from this document
        all_similar = self.vector_store.search_similar_sections(query, top_k=50)
        
        # Filter to only this document
        doc_sections = [
            (section_emb, score) for section_emb, score in all_similar
            if section_emb.doc_id == doc_id
        ]
        
        # Return top sections with content
        results = []
        for section_emb, score in doc_sections[:top_k]:
            results.append({
                'section_key': section_emb.section_key,
                'section_heading': section_emb.section_heading,
                'content_preview': section_emb.content_preview,
                'relevance_score': score
            })
        
        return results


if __name__ == "__main__":
    # Example usage
    vector_store = SimpleVectorStore(Path("vector_store"))
    analyzer = CriteriaAnalyzer(vector_store)
    
    print("Vector store and criteria analyzer ready!")
    print(f"Stats: {vector_store.get_stats()}")
