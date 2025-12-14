#!/usr/bin/env python3
"""
Results-Focused Web UI for Paper Analysis Pipeline
Shows actual fucking results with all 5 criteria for every paper
"""

import os
import sys
import json
import urllib.parse
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import webbrowser

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.markdown_parser import MarkdownParser, DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer
from src.core.analyzer import PaperAnalyzer

# Try to import LLM evaluator
try:
    from llm_evaluator import LLMPaperEvaluator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class ResultsHandler(BaseHTTPRequestHandler):
    """HTTP request handler focused on showing fucking results"""
    
    def __init__(self, *args, **kwargs):
        # Initialize components (shared across requests)
        if not hasattr(ResultsHandler, '_components_initialized'):
            self.setup_components()
            ResultsHandler._components_initialized = True
        super().__init__(*args, **kwargs)
    
    def setup_components(self):
        """Initialize analysis components"""
        print("🔧 Setting up analysis components...")
        
        db_path = Path("data/markdown_db")
        vector_path = Path("data/vector_store")
        
        ResultsHandler.parser = MarkdownParser()
        ResultsHandler.db = DocumentDatabase(db_path)
        ResultsHandler.vector_store = SimpleVectorStore(vector_path)
        ResultsHandler.criteria_analyzer = CriteriaAnalyzer(ResultsHandler.vector_store)
        ResultsHandler.regex_analyzer = PaperAnalyzer(ResultsHandler.db, ResultsHandler.vector_store)
        
        if LLM_AVAILABLE:
            ResultsHandler.llm_analyzer = LLMPaperEvaluator(ResultsHandler.db, ResultsHandler.vector_store)
        else:
            ResultsHandler.llm_analyzer = None
        
        print("✅ Components ready!")
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/':
                self.serve_main_page()
            elif self.path == '/api/stats':
                self.serve_stats()
            elif self.path.startswith('/api/analyze/'):
                parts = self.path.split('/')
                if len(parts) >= 5:
                    method = parts[3]
                    doc_id = parts[4]
                    self.serve_single_analysis(method, doc_id)
            else:
                self.send_error(404)
        except Exception as e:
            print(f"❌ Error in GET: {e}")
            self.send_error(500)
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path.startswith('/api/analyze_all/'):
                method = self.path.split('/')[-1]
                self.serve_analyze_all(method)
            else:
                self.send_error(404)
        except Exception as e:
            print(f"❌ Error in POST: {e}")
            self.send_error(500)
    
    def serve_main_page(self):
        """Serve the main results page"""
        # Load existing results
        regex_results = self.load_latest_results('regex')
        llm_results = self.load_latest_results('llm')
        
        html = self.generate_results_html(regex_results, llm_results)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def load_latest_results(self, method):
        """Load the latest results for a method"""
        try:
            # Find the most recent results file
            pattern = f"{method}_analysis_*.json"
            files = list(Path("archive/old_analysis_results").glob(pattern))
            if not files:
                return None
            
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {method} results: {e}")
            return None
    
    def generate_results_html(self, regex_results, llm_results):
        """Generate the main results HTML page"""
        
        # Count papers by recommendation
        def count_recommendations(results):
            if not results or 'results' not in results:
                return {'include': 0, 'exclude': 0, 'review': 0, 'total': 0}
            
            counts = {'include': 0, 'exclude': 0, 'review': 0}
            for paper in results['results']:
                rec = paper.get('recommendation', 'unknown').lower()
                if rec in counts:
                    counts[rec] += 1
            counts['total'] = len(results['results'])
            return counts
        
        regex_counts = count_recommendations(regex_results)
        llm_counts = count_recommendations(llm_results)
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>📊 Paper Analysis Results</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: #f8f9fa; }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #343a40; color: white; padding: 30px; text-align: center; border-radius: 8px; margin-bottom: 30px; }}
        
        /* Tabs */
        .tabs {{ display: flex; background: white; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .tab {{ flex: 1; padding: 20px; text-align: center; cursor: pointer; border: none; background: none; font-size: 18px; font-weight: bold; }}
        .tab.active {{ background: #007bff; color: white; }}
        .tab:hover {{ background: #e9ecef; }}
        .tab.active:hover {{ background: #0056b3; }}
        
        /* Tab Content */
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        /* Stats Cards */
        .stats {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ flex: 1; background: white; padding: 25px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-card.include {{ border-left: 5px solid #28a745; }}
        .stat-card.exclude {{ border-left: 5px solid #dc3545; }}
        .stat-card.review {{ border-left: 5px solid #ffc107; }}
        .stat-card.total {{ border-left: 5px solid #007bff; }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }}
        .stat-label {{ font-size: 1.1em; color: #666; }}
        
        /* Analysis Buttons */
        .actions {{ display: flex; gap: 15px; margin-bottom: 30px; }}
        .btn {{ padding: 15px 30px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; }}
        .btn-primary {{ background: #007bff; color: white; }}
        .btn-success {{ background: #28a745; color: white; }}
        .btn:hover {{ opacity: 0.9; transform: translateY(-1px); }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        
        /* Results Grid */
        .results-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }}
        .paper-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .paper-card.include {{ border-left: 5px solid #28a745; }}
        .paper-card.exclude {{ border-left: 5px solid #dc3545; }}
        .paper-card.review {{ border-left: 5px solid #ffc107; }}
        
        .paper-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; color: #333; }}
        .paper-score {{ font-size: 1.1em; margin-bottom: 15px; padding: 8px 12px; border-radius: 4px; display: inline-block; }}
        .paper-score.include {{ background: #d4edda; color: #155724; }}
        .paper-score.exclude {{ background: #f8d7da; color: #721c24; }}
        .paper-score.review {{ background: #fff3cd; color: #856404; }}
        
        /* Criteria Display */
        .criteria {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-top: 15px; }}
        .criterion {{ padding: 8px 12px; border-radius: 4px; text-align: center; font-size: 0.9em; font-weight: bold; }}
        .criterion.yes {{ background: #d4edda; color: #155724; }}
        .criterion.no {{ background: #f8d7da; color: #721c24; }}
        .criterion.unknown {{ background: #e2e3e5; color: #383d41; }}
        
        /* Loading */
        .loading {{ text-align: center; padding: 40px; color: #007bff; font-size: 1.2em; }}
        
        /* No Results */
        .no-results {{ text-align: center; padding: 40px; color: #666; }}
        .no-results h3 {{ margin-bottom: 20px; }}
        
        /* Progress */
        .progress-section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; display: none; }}
        .progress-bar {{ background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ background: #007bff; height: 100%; transition: width 0.3s; }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .results-grid {{ grid-template-columns: 1fr; }}
            .stats {{ flex-direction: column; }}
            .actions {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Paper Analysis Results</h1>
            <p>Deep Researcher Criteria Analysis - Real-time Results</p>
        </div>
        
        <!-- Analysis Actions -->
        <div class="actions">
            <button class="btn btn-success" onclick="runAnalysis('regex')">
                🔍 Run Regex Analysis
            </button>
            <button class="btn btn-primary" onclick="runAnalysis('llm')" {'disabled' if not LLM_AVAILABLE else ''}>
                🧠 Run LLM Analysis
            </button>
        </div>
        
        <!-- Progress Section -->
        <div id="progress-section" class="progress-section">
            <h3 id="progress-title">Analysis in Progress...</h3>
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
            </div>
            <div id="progress-text">Starting analysis...</div>
        </div>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab active" onclick="showTab('regex')">
                🔍 Regex Results ({regex_counts['total']} papers)
            </button>
            <button class="tab" onclick="showTab('llm')">
                🧠 LLM Results ({llm_counts['total']} papers)
            </button>
        </div>
        
        <!-- Regex Results Tab -->
        <div id="regex" class="tab-content active">
            {self.generate_method_results('Regex', regex_results, regex_counts)}
        </div>
        
        <!-- LLM Results Tab -->
        <div id="llm" class="tab-content">
            {self.generate_method_results('LLM', llm_results, llm_counts)}
        </div>
    </div>

    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
        
        function runAnalysis(method) {{
            const progressSection = document.getElementById('progress-section');
            const progressTitle = document.getElementById('progress-title');
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            
            progressSection.style.display = 'block';
            progressTitle.textContent = `Running ${{method.toUpperCase()}} Analysis...`;
            progressText.textContent = 'Starting analysis...';
            progressFill.style.width = '0%';
            
            fetch(`/api/analyze_all/${{method}}`, {{
                method: 'POST'
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.status === 'success') {{
                    progressFill.style.width = '100%';
                    progressText.textContent = 'Analysis complete! Reloading page...';
                    setTimeout(() => {{
                        window.location.reload();
                    }}, 1000);
                }} else {{
                    progressText.textContent = 'Error: ' + (data.message || 'Unknown error');
                }}
            }})
            .catch(error => {{
                progressText.textContent = 'Error: ' + error.message;
            }});
        }}
        
        // Auto-refresh every 30 seconds if no results
        const hasResults = {str(regex_results is not None or llm_results is not None).lower()};
        if (!hasResults) {{
            setTimeout(() => {{
                window.location.reload();
            }}, 30000);
        }}
    </script>
</body>
</html>"""
    
    def generate_method_results(self, method_name, results, counts):
        """Generate HTML for a specific method's results"""
        if not results or 'results' not in results:
            return f"""
            <div class="no-results">
                <h3>No {method_name} Results Available</h3>
                <p>Click "Run {method_name} Analysis" to generate results.</p>
            </div>
            """
        
        # Stats cards
        stats_html = f"""
        <div class="stats">
            <div class="stat-card total">
                <div class="stat-number">{counts['total']}</div>
                <div class="stat-label">Total Papers</div>
            </div>
            <div class="stat-card include">
                <div class="stat-number">{counts['include']}</div>
                <div class="stat-label">Include</div>
            </div>
            <div class="stat-card exclude">
                <div class="stat-number">{counts['exclude']}</div>
                <div class="stat-label">Exclude</div>
            </div>
            <div class="stat-card review">
                <div class="stat-number">{counts['review']}</div>
                <div class="stat-label">Review</div>
            </div>
        </div>
        """
        
        # Sort papers by score (highest first)
        papers = sorted(results['results'], key=lambda x: x.get('overall_score', 0), reverse=True)
        
        # Generate paper cards
        papers_html = '<div class="results-grid">'
        for paper in papers:
            papers_html += self.generate_paper_card(paper)
        papers_html += '</div>'
        
        return stats_html + papers_html
    
    def generate_paper_card(self, paper):
        """Generate HTML for a single paper card"""
        title = paper.get('title', 'Unknown Title')
        score = paper.get('overall_score', 0)
        recommendation = paper.get('recommendation', 'Unknown').lower()
        evaluations = paper.get('evaluations', {})
        
        # Format score
        score_text = f"{recommendation.title()} (Score: {score:.2f})"
        
        # Generate criteria badges
        criteria_html = ""
        criteria_order = ['pytorch', 'supervised', 'small_dataset', 'quick_training', 'has_repo']
        criteria_labels = {
            'pytorch': 'PyTorch',
            'supervised': 'Supervised',
            'small_dataset': 'Small Dataset',
            'quick_training': 'Quick Training',
            'has_repo': 'Has Repo'
        }
        
        for criterion in criteria_order:
            if criterion in evaluations:
                eval_data = evaluations[criterion]
                answer = eval_data.get('answer', 'Unknown').lower()
                confidence = eval_data.get('confidence', 0)
                
                # Determine CSS class
                css_class = 'unknown'
                if answer == 'yes':
                    css_class = 'yes'
                elif answer == 'no':
                    css_class = 'no'
                
                # Create badge with confidence info
                label = criteria_labels.get(criterion, criterion)
                if confidence > 0:
                    badge_text = f"{label}: {answer.title()} ({confidence:.1f})"
                else:
                    badge_text = f"{label}: {answer.title()}"
                
                criteria_html += f'<div class="criterion {css_class}">{badge_text}</div>'
        
        return f"""
        <div class="paper-card {recommendation}">
            <div class="paper-title">{title}</div>
            <div class="paper-score {recommendation}">{score_text}</div>
            <div class="criteria">
                {criteria_html}
            </div>
        </div>
        """
    
    def serve_analyze_all(self, method):
        """Run analysis on all papers"""
        try:
            print(f"🚀 Starting {method} analysis...")
            
            doc_list = self.db.list_documents()
            results = []
            
            for i, doc_info in enumerate(doc_list):
                doc_id = doc_info['doc_id']
                doc_title = doc_info['title']
                
                print(f"📄 Analyzing {i+1}/{len(doc_list)}: {doc_title}")
                
                try:
                    if method == 'llm' and self.llm_analyzer:
                        result = self.llm_analyzer.analyze_paper(doc_id)
                    else:
                        result = self.regex_analyzer.analyze_paper(doc_id)
                    
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error analyzing {doc_title}: {e}")
            
            # Save results
            if method == 'llm' and self.llm_analyzer:
                self.llm_analyzer.save_results(results)
            else:
                self.regex_analyzer.save_results(results)
            
            print(f"✅ {method} analysis complete! {len(results)} papers analyzed.")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'success', 'count': len(results)}
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'error', 'message': str(e)}
            self.wfile.write(json.dumps(response).encode())
    
    def serve_stats(self):
        """Serve system statistics"""
        stats = {
            'total_documents': len(self.db.list_documents()),
            'llm_available': LLM_AVAILABLE,
            'criteria_count': 5
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def main():
    """Start the results-focused web server"""
    port = 3444
    
    print(f"🚀 Starting Results Web UI on port {port}...")
    
    server = HTTPServer(('localhost', port), ResultsHandler)
    
    # Open browser
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open(f'http://localhost:{port}')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print(f"🌐 Results UI available at: http://localhost:{port}")
    print("📊 This UI shows ACTUAL RESULTS with all 5 criteria!")
    print("🔍 Regex analysis works perfectly - LLM has API issues")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
