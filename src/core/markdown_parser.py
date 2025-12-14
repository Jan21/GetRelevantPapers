"""
Markdown Document Parser

Parses markdown documents and extracts sections based on headings.
Each heading becomes a key and the content under it becomes the value.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime


@dataclass
class MarkdownSection:
    """A section extracted from markdown document"""
    heading: str
    content: str
    level: int  # Heading level (1-6)
    line_start: int
    line_end: int


@dataclass
class ParsedMarkdownDocument:
    """Parsed markdown document with sections"""
    file_path: str
    title: str
    sections: Dict[str, str]  # heading -> content
    section_objects: List[MarkdownSection]  # Full section objects
    raw_content: str
    parsed_at: datetime
    file_hash: str


class MarkdownParser:
    """Parse markdown documents into structured sections"""
    
    def __init__(self):
        # Regex pattern to match markdown headings
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def parse_file(self, file_path: Path) -> ParsedMarkdownDocument:
        """Parse a markdown file and extract sections"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        content = file_path.read_text(encoding='utf-8')
        return self.parse_content(content, str(file_path))
    
    def parse_content(self, content: str, file_path: str = "unknown") -> ParsedMarkdownDocument:
        """Parse markdown content and extract sections"""
        
        # Calculate file hash for change detection
        file_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Split content into lines for processing
        lines = content.split('\n')
        
        # Find all headings
        headings = []
        for i, line in enumerate(lines):
            match = self.heading_pattern.match(line)
            if match:
                level = len(match.group(1))  # Count # symbols
                heading_text = match.group(2).strip()
                headings.append((i, level, heading_text))
        
        # Extract sections
        sections = {}
        section_objects = []
        
        # Extract document title (first H1 heading or filename)
        title = Path(file_path).stem
        if headings and headings[0][1] == 1:  # First heading is H1
            title = headings[0][2]
        
        for i, (line_num, level, heading_text) in enumerate(headings):
            # Find the end of this section (next heading of same or higher level)
            section_end = len(lines)
            for j in range(i + 1, len(headings)):
                next_line, next_level, _ = headings[j]
                if next_level <= level:  # Same or higher level heading
                    section_end = next_line
                    break
            
            # Extract content between headings
            content_lines = lines[line_num + 1:section_end]
            section_content = '\n'.join(content_lines).strip()
            
            # Clean heading text for use as key
            clean_heading = self._clean_heading_key(heading_text)
            
            # Store section
            sections[clean_heading] = section_content
            
            section_objects.append(MarkdownSection(
                heading=heading_text,
                content=section_content,
                level=level,
                line_start=line_num + 1,
                line_end=section_end
            ))
        
        return ParsedMarkdownDocument(
            file_path=file_path,
            title=title,
            sections=sections,
            section_objects=section_objects,
            raw_content=content,
            parsed_at=datetime.now(),
            file_hash=file_hash
        )
    
    def _clean_heading_key(self, heading: str) -> str:
        """Clean heading text to create a consistent key"""
        # Remove special characters, convert to lowercase, replace spaces with underscores
        cleaned = re.sub(r'[^\w\s-]', '', heading.lower())
        cleaned = re.sub(r'\s+', '_', cleaned.strip())
        return cleaned
    
    def parse_directory(self, directory: Path, pattern: str = "*.md") -> List[ParsedMarkdownDocument]:
        """Parse all markdown files in a directory"""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        markdown_files = list(directory.glob(pattern))
        parsed_docs = []
        
        for file_path in markdown_files:
            try:
                doc = self.parse_file(file_path)
                parsed_docs.append(doc)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                continue
        
        return parsed_docs


class DocumentDatabase:
    """Simple disk-based document database for parsed markdown documents"""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Database structure:
        # db_path/
        #   documents.json  # Document metadata
        #   sections/       # Individual section files
        #   raw/           # Raw document content
        
        self.documents_file = self.db_path / "documents.json"
        self.sections_dir = self.db_path / "sections"
        self.raw_dir = self.db_path / "raw"
        
        self.sections_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        
        # Load existing documents
        self.documents = self._load_documents()
    
    def _load_documents(self) -> Dict[str, Dict]:
        """Load document metadata from disk"""
        if self.documents_file.exists():
            try:
                with open(self.documents_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_documents(self):
        """Save document metadata to disk"""
        with open(self.documents_file, 'w') as f:
            json.dump(self.documents, f, indent=2, default=str)
    
    def _get_doc_id(self, file_path: str) -> str:
        """Generate document ID from file path"""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def store_document(self, doc: ParsedMarkdownDocument) -> str:
        """Store a parsed document in the database"""
        doc_id = self._get_doc_id(doc.file_path)
        
        # Check if document has changed
        if doc_id in self.documents:
            if self.documents[doc_id]['file_hash'] == doc.file_hash:
                print(f"Document {doc.file_path} unchanged, skipping")
                return doc_id
        
        # Store document metadata
        self.documents[doc_id] = {
            'file_path': doc.file_path,
            'title': doc.title,
            'file_hash': doc.file_hash,
            'parsed_at': doc.parsed_at.isoformat(),
            'section_count': len(doc.sections),
            'section_keys': list(doc.sections.keys())
        }
        
        # Store raw content
        raw_file = self.raw_dir / f"{doc_id}.txt"
        raw_file.write_text(doc.raw_content, encoding='utf-8')
        
        # Store sections individually
        sections_file = self.sections_dir / f"{doc_id}.json"
        sections_data = {
            'sections': doc.sections,
            'section_objects': [
                {
                    'heading': s.heading,
                    'content': s.content,
                    'level': s.level,
                    'line_start': s.line_start,
                    'line_end': s.line_end
                }
                for s in doc.section_objects
            ]
        }
        
        with open(sections_file, 'w') as f:
            json.dump(sections_data, f, indent=2)
        
        # Save metadata
        self._save_documents()
        
        print(f"Stored document: {doc.title} ({len(doc.sections)} sections)")
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata by ID"""
        return self.documents.get(doc_id)
    
    def get_document_sections(self, doc_id: str) -> Optional[Dict[str, str]]:
        """Get document sections by ID"""
        sections_file = self.sections_dir / f"{doc_id}.json"
        if sections_file.exists():
            with open(sections_file, 'r') as f:
                data = json.load(f)
                return data.get('sections', {})
        return None
    
    def get_document_raw(self, doc_id: str) -> Optional[str]:
        """Get raw document content by ID"""
        raw_file = self.raw_dir / f"{doc_id}.txt"
        if raw_file.exists():
            return raw_file.read_text(encoding='utf-8')
        return None
    
    def search_sections(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple text search across all sections"""
        results = []
        query_lower = query.lower()
        
        for doc_id, doc_meta in self.documents.items():
            sections = self.get_document_sections(doc_id)
            if not sections:
                continue
            
            for section_key, content in sections.items():
                if query_lower in content.lower() or query_lower in section_key.lower():
                    results.append({
                        'doc_id': doc_id,
                        'document_title': doc_meta['title'],
                        'section_key': section_key,
                        'content': content[:500] + '...' if len(content) > 500 else content,
                        'relevance_score': self._calculate_relevance(query_lower, section_key, content)
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:limit]
    
    def _calculate_relevance(self, query: str, section_key: str, content: str) -> float:
        """Calculate simple relevance score"""
        score = 0.0
        
        # Exact matches in section key get high score
        if query in section_key.lower():
            score += 10.0
        
        # Count occurrences in content
        content_lower = content.lower()
        occurrences = content_lower.count(query)
        score += occurrences * 1.0
        
        # Bonus for query words
        query_words = query.split()
        for word in query_words:
            if word in section_key.lower():
                score += 5.0
            score += content_lower.count(word) * 0.5
        
        return score
    
    def get_all_section_keys(self) -> List[Tuple[str, str, str]]:
        """Get all section keys across all documents"""
        section_keys = []
        
        for doc_id, doc_meta in self.documents.items():
            for section_key in doc_meta.get('section_keys', []):
                section_keys.append((doc_id, doc_meta['title'], section_key))
        
        return section_keys
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the database"""
        return [
            {
                'doc_id': doc_id,
                'title': meta['title'],
                'file_path': meta['file_path'],
                'section_count': meta['section_count'],
                'parsed_at': meta['parsed_at']
            }
            for doc_id, meta in self.documents.items()
        ]


if __name__ == "__main__":
    # Example usage
    parser = MarkdownParser()
    db = DocumentDatabase(Path("markdown_db"))
    
    # Parse a single file
    # doc = parser.parse_file(Path("example.md"))
    # doc_id = db.store_document(doc)
    
    # Parse directory
    # docs = parser.parse_directory(Path("markdown_files"))
    # for doc in docs:
    #     db.store_document(doc)
    
    # Search sections
    # results = db.search_sections("methodology")
    # for result in results:
    #     print(f"{result['document_title']} - {result['section_key']}")
    
    print("Markdown parser and database ready!")
