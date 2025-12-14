#!/usr/bin/env python3
"""
Simple Web UI for Markdown Paper Analysis Pipeline

A Flask-based web interface for uploading papers, running analysis,
and viewing results.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

try:
    from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

from src.core.markdown_parser import MarkdownParser, DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer
from src.core.analyzer import PaperAnalyzer
from txt_to_markdown import TxtToMarkdownConverter

# Try to import LLM evaluator
try:
    from llm_evaluator import LLMPaperEvaluator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class PaperAnalysisUI:
    """Web UI for paper analysis pipeline"""
    
    def __init__(self):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the web UI")
        
        self.app = Flask(__name__)
        self.app.secret_key = 'paper-analysis-ui-secret-key'
        
        # Initialize components
        self.setup_components()
        self.setup_routes()
        
        # Configuration
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.upload_folder = Path("uploads")
        self.upload_folder.mkdir(exist_ok=True)
        
        # Allowed file extensions
        self.allowed_extensions = {'.md', '.txt', '.pdf'}
    
    def setup_components(self):
        """Initialize analysis components"""
        self.db_path = Path("data/markdown_db")
        self.vector_path = Path("data/vector_store")
        
        self.parser = MarkdownParser()
        self.db = DocumentDatabase(self.db_path)
        self.vector_store = SimpleVectorStore(self.vector_path)
        self.criteria_analyzer = CriteriaAnalyzer(self.vector_store)
        
        # Initialize analyzers
        self.regex_analyzer = PaperAnalyzer(self.db, self.vector_store)
        
        if LLM_AVAILABLE:
            self.llm_analyzer = LLMPaperEvaluator(self.db, self.vector_store)
        else:
            self.llm_analyzer = None
        
        self.txt_converter = TxtToMarkdownConverter()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            stats = self.get_system_stats()
            recent_papers = self.get_recent_papers()
            return render_template('index.html', stats=stats, recent_papers=recent_papers)
        
        @self.app.route('/upload', methods=['GET', 'POST'])
        def upload_paper():
            """Upload and process papers"""
            if request.method == 'POST':
                return self.handle_upload()
            return render_template('upload.html')
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_papers():
            """Analyze papers with selected method"""
            method = request.form.get('method', 'regex')
            paper_ids = request.form.getlist('paper_ids')
            
            if not paper_ids:
                paper_ids = None  # Analyze all papers
            
            results = self.run_analysis(method, paper_ids)
            return jsonify(results)
        
        @self.app.route('/results')
        def view_results():
            """View analysis results"""
            # Get latest results
            results_files = list(Path("archive/old_analysis_results").glob("*_analysis.json"))
            if results_files:
                latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    results = json.load(f)
            else:
                results = {"results": [], "summary": {}}
            
            return render_template('results.html', results=results)
        
        @self.app.route('/papers')
        def list_papers():
            """List all papers in database"""
            papers = self.db.list_documents()
            return render_template('papers.html', papers=papers)
        
        @self.app.route('/paper/<doc_id>')
        def view_paper(doc_id):
            """View individual paper details"""
            doc_meta = self.db.get_document(doc_id)
            sections = self.db.get_document_sections(doc_id)
            
            if not doc_meta:
                flash('Paper not found', 'error')
                return redirect(url_for('list_papers'))
            
            return render_template('paper_detail.html', 
                                 doc_meta=doc_meta, 
                                 sections=sections,
                                 doc_id=doc_id)
        
        @self.app.route('/api/stats')
        def api_stats():
            """API endpoint for system statistics"""
            return jsonify(self.get_system_stats())
        
        @self.app.route('/api/analyze/<doc_id>/<method>')
        def api_analyze_single(doc_id, method):
            """API endpoint to analyze single paper"""
            if method == 'llm' and not self.llm_analyzer:
                return jsonify({"error": "LLM analyzer not available"}), 400
            
            analyzer = self.llm_analyzer if method == 'llm' else self.regex_analyzer
            result = analyzer.analyze_paper(doc_id)
            
            if result:
                return jsonify({
                    "doc_id": result.doc_id,
                    "title": result.title,
                    "overall_score": result.overall_score,
                    "recommendation": result.recommendation,
                    "evaluations": {
                        name: {
                            "answer": eval.answer,
                            "confidence": eval.confidence,
                            "evidence": eval.evidence
                        }
                        for name, eval in result.evaluations.items()
                    }
                })
            else:
                return jsonify({"error": "Analysis failed"}), 500
    
    def handle_upload(self):
        """Handle file upload"""
        if 'files' not in request.files:
            flash('No files selected', 'error')
            return redirect(request.url)
        
        files = request.files.getlist('files')
        processed_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = self.upload_folder / filename
                file.save(file_path)
                
                # Process the file
                try:
                    doc_id = self.process_uploaded_file(file_path)
                    if doc_id:
                        processed_files.append({
                            'filename': filename,
                            'doc_id': doc_id,
                            'status': 'success'
                        })
                    else:
                        processed_files.append({
                            'filename': filename,
                            'status': 'error',
                            'message': 'Processing failed'
                        })
                except Exception as e:
                    processed_files.append({
                        'filename': filename,
                        'status': 'error',
                        'message': str(e)
                    })
        
        if processed_files:
            flash(f'Processed {len(processed_files)} files', 'success')
        
        return render_template('upload_results.html', results=processed_files)
    
    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return Path(filename).suffix.lower() in self.allowed_extensions
    
    def process_uploaded_file(self, file_path: Path) -> Optional[str]:
        """Process uploaded file and add to database"""
        if file_path.suffix.lower() == '.md':
            # Direct markdown file
            doc = self.parser.parse_file(file_path)
            doc_id = self.db.store_document(doc)
            
            # Add to vector store
            sections_data = []
            for section_key, content in doc.sections.items():
                sections_data.append({
                    'doc_id': doc_id,
                    'document_title': doc.title,
                    'section_key': section_key,
                    'section_heading': section_key.replace('_', ' ').title(),
                    'content': content
                })
            
            if sections_data:
                self.vector_store.add_sections(sections_data)
            
            return doc_id
        
        elif file_path.suffix.lower() == '.txt':
            # Convert TXT to markdown first
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            title = file_path.stem.replace('_', ' ')
            
            markdown_content = self.txt_converter.convert_to_markdown(content, title)
            
            # Create temporary markdown file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
                tmp_file.write(markdown_content)
                tmp_path = Path(tmp_file.name)
            
            try:
                doc = self.parser.parse_file(tmp_path)
                doc_id = self.db.store_document(doc)
                
                # Add to vector store
                sections_data = []
                for section_key, content in doc.sections.items():
                    sections_data.append({
                        'doc_id': doc_id,
                        'document_title': doc.title,
                        'section_key': section_key,
                        'section_heading': section_key.replace('_', ' ').title(),
                        'content': content
                    })
                
                if sections_data:
                    self.vector_store.add_sections(sections_data)
                
                return doc_id
            finally:
                tmp_path.unlink()  # Clean up temp file
        
        return None
    
    def run_analysis(self, method: str, paper_ids: Optional[List[str]] = None) -> Dict:
        """Run analysis with specified method"""
        if method == 'llm' and not self.llm_analyzer:
            return {"error": "LLM analyzer not available"}
        
        analyzer = self.llm_analyzer if method == 'llm' else self.regex_analyzer
        
        if paper_ids:
            # Analyze specific papers
            results = []
            for doc_id in paper_ids:
                result = analyzer.analyze_paper(doc_id)
                if result:
                    results.append(result)
        else:
            # Analyze all papers
            if hasattr(analyzer, 'analyze_all_papers'):
                results = analyzer.analyze_all_papers()
            else:
                results = analyzer.analyze_all()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"{method}_analysis_{timestamp}.json")
        analyzer.save_results(results, output_file)
        
        # Return summary
        return {
            "method": method,
            "total_papers": len(results),
            "include_count": sum(1 for r in results if r.recommendation == "Include"),
            "exclude_count": sum(1 for r in results if r.recommendation == "Exclude"),
            "review_count": sum(1 for r in results if r.recommendation == "Review"),
            "output_file": str(output_file),
            "results": [
                {
                    "title": r.title,
                    "score": r.overall_score,
                    "recommendation": r.recommendation
                }
                for r in results[:10]  # First 10 results
            ]
        }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_documents": len(self.db.documents),
            "vector_embeddings": self.vector_store.get_stats()["total_embeddings"],
            "criteria_count": len(self.regex_analyzer.criteria),
            "llm_available": LLM_AVAILABLE,
            "database_path": str(self.db_path),
            "vector_store_path": str(self.vector_path)
        }
    
    def get_recent_papers(self, limit: int = 5) -> List[Dict]:
        """Get recently added papers"""
        papers = self.db.list_documents()
        # Sort by parsed_at if available
        sorted_papers = sorted(papers, 
                             key=lambda p: p.get('parsed_at', ''), 
                             reverse=True)
        return sorted_papers[:limit]
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the Flask application"""
        print(f"🚀 Starting Paper Analysis UI...")
        print(f"📊 System Stats: {self.get_system_stats()}")
        print(f"🌐 Access at: http://{host}:{port}")
        
        self.app.run(host=host, port=port, debug=debug)


# HTML Templates (embedded for simplicity)
def create_templates():
    """Create HTML templates"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Paper Analysis Pipeline{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-file-alt"></i> Paper Analysis
            </a>
            <div class="navbar-nav">
                <a class="nav-link" href="{{ url_for('index') }}">Dashboard</a>
                <a class="nav-link" href="{{ url_for('upload_paper') }}">Upload</a>
                <a class="nav-link" href="{{ url_for('list_papers') }}">Papers</a>
                <a class="nav-link" href="{{ url_for('view_results') }}">Results</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else 'success' }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    # Index template
    index_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <h1><i class="fas fa-tachometer-alt"></i> Dashboard</h1>
        <p class="lead">Markdown Paper Analysis Pipeline</p>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card bg-primary text-white">
                    <div class="card-body">
                        <h5>{{ stats.total_documents }}</h5>
                        <small>Documents</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-success text-white">
                    <div class="card-body">
                        <h5>{{ stats.vector_embeddings }}</h5>
                        <small>Embeddings</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white">
                    <div class="card-body">
                        <h5>{{ stats.criteria_count }}</h5>
                        <small>Criteria</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-white">
                    <div class="card-body">
                        <h5>{{ 'Yes' if stats.llm_available else 'No' }}</h5>
                        <small>LLM Available</small>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="mt-4">
            <h3>Quick Actions</h3>
            <div class="btn-group" role="group">
                <a href="{{ url_for('upload_paper') }}" class="btn btn-primary">
                    <i class="fas fa-upload"></i> Upload Papers
                </a>
                <button class="btn btn-success" onclick="analyzeAll('regex')">
                    <i class="fas fa-search"></i> Analyze (Regex)
                </button>
                {% if stats.llm_available %}
                <button class="btn btn-info" onclick="analyzeAll('llm')">
                    <i class="fas fa-brain"></i> Analyze (LLM)
                </button>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <h3>Recent Papers</h3>
        <div class="list-group">
            {% for paper in recent_papers %}
            <a href="{{ url_for('view_paper', doc_id=paper.doc_id) }}" class="list-group-item list-group-item-action">
                <h6 class="mb-1">{{ paper.title[:50] }}...</h6>
                <small>{{ paper.section_count }} sections</small>
            </a>
            {% endfor %}
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
function analyzeAll(method) {
    fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: 'method=' + method
    })
    .then(response => response.json())
    .then(data => {
        alert('Analysis complete! ' + data.total_papers + ' papers analyzed.');
        window.location.href = '/results';
    })
    .catch(error => {
        alert('Analysis failed: ' + error);
    });
}
</script>
{% endblock %}'''
    
    # Upload template
    upload_template = '''{% extends "base.html" %}

{% block content %}
<h1><i class="fas fa-upload"></i> Upload Papers</h1>

<form method="POST" enctype="multipart/form-data" class="mt-4">
    <div class="mb-3">
        <label for="files" class="form-label">Select Papers (.md, .txt, .pdf)</label>
        <input type="file" class="form-control" name="files" multiple accept=".md,.txt,.pdf">
        <div class="form-text">You can select multiple files. Supported formats: Markdown (.md), Text (.txt), PDF (.pdf)</div>
    </div>
    
    <button type="submit" class="btn btn-primary">
        <i class="fas fa-upload"></i> Upload and Process
    </button>
</form>

<div class="mt-4">
    <h3>Supported Formats</h3>
    <ul>
        <li><strong>Markdown (.md)</strong>: Direct processing</li>
        <li><strong>Text (.txt)</strong>: Converted to markdown</li>
        <li><strong>PDF (.pdf)</strong>: Extract text and convert (coming soon)</li>
    </ul>
</div>
{% endblock %}'''
    
    # Write templates
    (templates_dir / "base.html").write_text(base_template)
    (templates_dir / "index.html").write_text(index_template)
    (templates_dir / "upload.html").write_text(upload_template)
    
    print("✓ HTML templates created")


if __name__ == "__main__":
    if not FLASK_AVAILABLE:
        print("Flask is required for the web UI. Install with: pip install flask")
        exit(1)
    
    # Create templates
    create_templates()
    
    # Initialize and run UI
    ui = PaperAnalysisUI()
    ui.run(debug=True)
