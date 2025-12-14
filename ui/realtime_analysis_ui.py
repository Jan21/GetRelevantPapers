#!/usr/bin/env python3
"""
Real-time Paper Analysis UI with Individual Paper Progress
Shows fucking progress for each paper as it's being analyzed
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
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.markdown_parser import MarkdownParser, DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer
from src.core.analyzer import PaperAnalyzer

# Try to import FREE LLM evaluator
try:
    from simple_free_llm import SimpleFreeEvaluator
    FREE_LLM_AVAILABLE = True
except ImportError:
    FREE_LLM_AVAILABLE = False


class RealtimeAnalysisHandler(BaseHTTPRequestHandler):
    """HTTP request handler with real-time analysis progress"""
    
    # Shared state for real-time updates
    analysis_status = {
        'running': False,
        'method': None,
        'current_paper': None,
        'progress': 0,
        'total': 0,
        'results': [],
        'errors': []
    }
    
    def __init__(self, *args, **kwargs):
        # Initialize components (shared across requests)
        if not hasattr(RealtimeAnalysisHandler, '_components_initialized'):
            self.setup_components()
            RealtimeAnalysisHandler._components_initialized = True
        super().__init__(*args, **kwargs)
    
    def setup_components(self):
        """Initialize analysis components"""
        print("🔧 Setting up analysis components...")
        
        db_path = Path("data/markdown_db")
        vector_path = Path("data/vector_store")
        
        RealtimeAnalysisHandler.parser = MarkdownParser()
        RealtimeAnalysisHandler.db = DocumentDatabase(db_path)
        RealtimeAnalysisHandler.vector_store = SimpleVectorStore(vector_path)
        RealtimeAnalysisHandler.criteria_analyzer = CriteriaAnalyzer(RealtimeAnalysisHandler.vector_store)
        RealtimeAnalysisHandler.regex_analyzer = PaperAnalyzer(RealtimeAnalysisHandler.db, RealtimeAnalysisHandler.vector_store)
        
        if FREE_LLM_AVAILABLE:
            RealtimeAnalysisHandler.llm_analyzer = SimpleFreeEvaluator(RealtimeAnalysisHandler.db, RealtimeAnalysisHandler.vector_store)
        else:
            RealtimeAnalysisHandler.llm_analyzer = None
        
        print("✅ Components ready!")
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/':
                self.serve_main_page()
            elif self.path == '/api/status':
                self.serve_analysis_status()
            elif self.path == '/api/results':
                self.serve_latest_results()
            else:
                self.send_error(404)
        except Exception as e:
            print(f"❌ Error in GET: {e}")
            self.send_error(500)
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path.startswith('/api/analyze/'):
                method = self.path.split('/')[-1]
                self.start_analysis(method)
            else:
                self.send_error(404)
        except Exception as e:
            print(f"❌ Error in POST: {e}")
            self.send_error(500)
    
    def serve_main_page(self):
        """Serve the main analysis page"""
        html = self.generate_main_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_analysis_status(self):
        """Serve current analysis status"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(self.analysis_status).encode())
    
    def serve_latest_results(self):
        """Serve the latest analysis results (truncated for browser performance)"""
        try:
            # Find the most recent results files
            regex_files = list(Path("archive/old_analysis_results").glob("regex_analysis_*.json"))
            llm_files = list(Path("archive/old_analysis_results").glob("free_llm_analysis_*.json")) + list(Path("archive/old_analysis_results").glob("llm_analysis_*.json"))
            
            results = {
                'regex': None,
                'llm': None
            }
            
            if regex_files:
                latest_regex = max(regex_files, key=lambda f: f.stat().st_mtime)
                with open(latest_regex, 'r') as f:
                    data = json.load(f)
                    # Truncate evidence strings to prevent browser freezing
                    if 'results' in data:
                        for paper in data['results']:
                            if 'evaluations' in paper:
                                for criterion, eval_data in paper['evaluations'].items():
                                    if 'evidence' in eval_data and len(eval_data['evidence']) > 200:
                                        eval_data['evidence'] = eval_data['evidence'][:200] + "..."
                    results['regex'] = data
            
            if llm_files:
                latest_llm = max(llm_files, key=lambda f: f.stat().st_mtime)
                with open(latest_llm, 'r') as f:
                    data = json.load(f)
                    # Truncate evidence strings to prevent browser freezing
                    if 'results' in data:
                        for paper in data['results']:
                            if 'evaluations' in paper:
                                for criterion, eval_data in paper['evaluations'].items():
                                    if 'evidence' in eval_data and len(eval_data['evidence']) > 200:
                                        eval_data['evidence'] = eval_data['evidence'][:200] + "..."
                    results['llm'] = data
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(results).encode())
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'error': str(e)}
            self.wfile.write(json.dumps(response).encode())
    
    def start_analysis(self, method):
        """Start analysis in background thread"""
        if self.analysis_status['running']:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'error', 'message': 'Analysis already running'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Start analysis in background
        analysis_thread = threading.Thread(target=self.run_analysis, args=(method,))
        analysis_thread.daemon = True
        analysis_thread.start()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {'status': 'started', 'method': method}
        self.wfile.write(json.dumps(response).encode())
    
    def run_analysis(self, method):
        """Run the actual analysis with real-time updates"""
        try:
            print(f"🚀 Starting {method} analysis...")
            
            # Reset status
            self.analysis_status.update({
                'running': True,
                'method': method,
                'current_paper': None,
                'progress': 0,
                'total': 0,
                'results': [],
                'errors': []
            })
            
            doc_list = self.db.list_documents()
            self.analysis_status['total'] = len(doc_list)
            
            results = []
            
            for i, doc_info in enumerate(doc_list):
                doc_id = doc_info['doc_id']
                doc_title = doc_info['title']
                
                # Update status
                self.analysis_status.update({
                    'current_paper': doc_title,
                    'progress': i + 1
                })
                
                print(f"📄 Analyzing {i+1}/{len(doc_list)}: {doc_title}")
                
                try:
                    start_time = time.time()
                    
                    if method == 'llm' and self.llm_analyzer:
                        result = self.llm_analyzer.analyze_paper(doc_id)
                    else:
                        result = self.regex_analyzer.analyze_paper(doc_id)
                    
                    end_time = time.time()
                    
                    # Add result to real-time status
                    result_dict = result.to_dict()
                    result_dict['analysis_time'] = round(end_time - start_time, 2)
                    self.analysis_status['results'].append(result_dict)
                    
                    results.append(result)
                    
                    print(f"✅ Completed {doc_title} in {result_dict['analysis_time']}s - {result.recommendation}")
                    
                except Exception as e:
                    error_msg = f"Error analyzing {doc_title}: {str(e)}"
                    print(f"❌ {error_msg}")
                    self.analysis_status['errors'].append(error_msg)
            
            # Save results
            if method == 'llm' and self.llm_analyzer:
                self.llm_analyzer.save_results(results)
            else:
                self.regex_analyzer.save_results(results)
            
            print(f"✅ {method} analysis complete! {len(results)} papers analyzed.")
            
            # Mark as complete
            self.analysis_status['running'] = False
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            self.analysis_status.update({
                'running': False,
                'errors': [f"Analysis failed: {str(e)}"]
            })
    
    def generate_main_html(self):
        """Generate the main HTML page"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>🔬 Real-time Paper Analysis</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: #f8f9fa; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #343a40; color: white; padding: 30px; text-align: center; border-radius: 8px; margin-bottom: 30px; }}
        
        /* Control Panel */
        .controls {{ background: white; padding: 25px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .btn {{ padding: 15px 30px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; margin-right: 15px; }}
        .btn-primary {{ background: #007bff; color: white; }}
        .btn-success {{ background: #28a745; color: white; }}
        .btn-danger {{ background: #dc3545; color: white; }}
        .btn:hover {{ opacity: 0.9; transform: translateY(-1px); }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none; }}
        
        /* Progress Section */
        .progress-section {{ background: white; padding: 25px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none; }}
        .progress-section.active {{ display: block; }}
        .progress-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .progress-bar {{ background: #e9ecef; height: 25px; border-radius: 12px; overflow: hidden; margin-bottom: 15px; }}
        .progress-fill {{ background: #007bff; height: 100%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }}
        .current-paper {{ font-size: 1.1em; color: #007bff; font-weight: bold; }}
        
        /* Real-time Results */
        .realtime-results {{ background: white; padding: 25px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none; }}
        .realtime-results.active {{ display: block; }}
        .result-item {{ padding: 15px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }}
        .result-item:last-child {{ border-bottom: none; }}
        .result-item.include {{ border-left: 4px solid #28a745; }}
        .result-item.exclude {{ border-left: 4px solid #dc3545; }}
        .result-item.review {{ border-left: 4px solid #ffc107; }}
        .result-title {{ font-weight: bold; flex: 1; }}
        .result-score {{ margin: 0 15px; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
        .result-score.include {{ background: #d4edda; color: #155724; }}
        .result-score.exclude {{ background: #f8d7da; color: #721c24; }}
        .result-score.review {{ background: #fff3cd; color: #856404; }}
        .result-time {{ color: #666; font-size: 0.9em; }}
        
        /* Final Results */
        .final-results {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none; }}
        .final-results.active {{ display: block; }}
        .results-tabs {{ display: flex; margin-bottom: 20px; }}
        .results-tab {{ padding: 15px 25px; cursor: pointer; border: 1px solid #dee2e6; background: #f8f9fa; }}
        .results-tab.active {{ background: #007bff; color: white; border-color: #007bff; }}
        .results-tab:first-child {{ border-radius: 6px 0 0 6px; }}
        .results-tab:last-child {{ border-radius: 0 6px 6px 0; }}
        
        .results-content {{ display: none; }}
        .results-content.active {{ display: block; }}
        
        /* Stats */
        .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .stat-card {{ flex: 1; padding: 20px; border-radius: 6px; text-align: center; }}
        .stat-card.total {{ background: #e3f2fd; color: #1565c0; }}
        .stat-card.include {{ background: #e8f5e8; color: #2e7d32; }}
        .stat-card.exclude {{ background: #ffebee; color: #c62828; }}
        .stat-card.review {{ background: #fff8e1; color: #ef6c00; }}
        .stat-number {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 1.1em; }}
        
        /* Papers Grid */
        .papers-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 15px; }}
        .paper-card {{ padding: 20px; border-radius: 6px; border-left: 4px solid #007bff; }}
        .paper-card.include {{ border-left-color: #28a745; background: #f8fff8; }}
        .paper-card.exclude {{ border-left-color: #dc3545; background: #fff8f8; }}
        .paper-card.review {{ border-left-color: #ffc107; background: #fffef8; }}
        .paper-title {{ font-weight: bold; margin-bottom: 10px; }}
        .paper-score {{ font-size: 1.1em; margin-bottom: 15px; }}
        .criteria {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .criterion {{ padding: 4px 8px; border-radius: 4px; font-size: 0.85em; }}
        .criterion.yes {{ background: #d4edda; color: #155724; }}
        .criterion.no {{ background: #f8d7da; color: #721c24; }}
        .criterion.unknown {{ background: #e2e3e5; color: #383d41; }}
        
        /* Loading */
        .loading {{ text-align: center; padding: 40px; color: #007bff; }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .stats {{ flex-direction: column; }}
            .papers-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Real-time Paper Analysis</h1>
            <p>Deep Researcher Criteria Analysis - Live Progress Tracking</p>
        </div>
        
        <!-- Control Panel -->
        <div class="controls">
            <h3>📊 Analysis Controls</h3>
            <button id="regex-btn" class="btn btn-success" onclick="startAnalysis('regex')">
                🔍 Run Regex Analysis (Fast)
            </button>
            <button id="llm-btn" class="btn btn-primary" onclick="startAnalysis('llm')" {'disabled' if not LLM_AVAILABLE else ''}>
                🧠 Run FREE LLM Analysis (Local Ollama)
            </button>
            <button id="stop-btn" class="btn btn-danger" onclick="stopAnalysis()" style="display: none;">
                ⏹️ Stop Analysis
            </button>
            <button id="view-results-btn" class="btn btn-primary" onclick="loadResults()" style="display: none;">
                📋 View Latest Results
            </button>
        </div>
        
        <!-- Progress Section -->
        <div id="progress-section" class="progress-section">
            <div class="progress-header">
                <h3 id="progress-title">🚀 Analysis in Progress...</h3>
                <span id="progress-counter">0 / 0</span>
            </div>
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill" style="width: 0%;">0%</div>
            </div>
            <div id="current-paper" class="current-paper">Starting analysis...</div>
        </div>
        
        <!-- Real-time Results -->
        <div id="realtime-results" class="realtime-results">
            <h3>📈 Live Results</h3>
            <div id="results-list"></div>
        </div>
        
        <!-- Final Results -->
        <div id="final-results" class="final-results">
            <h3>📋 Analysis Results</h3>
            <div class="results-tabs">
                <div class="results-tab active" onclick="showResultsTab('regex')">🔍 Regex Results</div>
                <div class="results-tab" onclick="showResultsTab('llm')">🧠 LLM Results</div>
            </div>
            
            <div id="regex-results" class="results-content active">
                <div class="loading">No regex results available. Run regex analysis first.</div>
            </div>
            
            <div id="llm-results" class="results-content">
                <div class="loading">No LLM results available. Run LLM analysis first.</div>
            </div>
        </div>
    </div>

    <script>
        let analysisInterval = null;
        let isAnalysisRunning = false;
        
        function startAnalysis(method) {{
            if (isAnalysisRunning) return;
            
            // Reset UI
            document.getElementById('progress-section').classList.add('active');
            document.getElementById('realtime-results').classList.add('active');
            document.getElementById('final-results').classList.remove('active');
            document.getElementById('results-list').innerHTML = '';
            
            // Update buttons
            document.getElementById('regex-btn').disabled = true;
            document.getElementById('llm-btn').disabled = true;
            document.getElementById('stop-btn').style.display = 'inline-block';
            document.getElementById('view-results-btn').style.display = 'none';
            
            isAnalysisRunning = true;
            
            // Start analysis
            fetch(`/api/analyze/${{method}}`, {{ method: 'POST' }})
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'started') {{
                        startProgressPolling();
                    }} else {{
                        alert('Failed to start analysis: ' + (data.message || 'Unknown error'));
                        resetUI();
                    }}
                }})
                .catch(error => {{
                    alert('Error starting analysis: ' + error.message);
                    resetUI();
                }});
        }}
        
        function startProgressPolling() {{
            analysisInterval = setInterval(() => {{
                fetch('/api/status')
                    .then(response => response.json())
                    .then(status => {{
                        updateProgress(status);
                        
                        if (!status.running && isAnalysisRunning) {{
                            // Analysis completed
                            clearInterval(analysisInterval);
                            analysisCompleted();
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error polling status:', error);
                    }});
            }}, 1000); // Poll every second
        }}
        
        function updateProgress(status) {{
            const progressFill = document.getElementById('progress-fill');
            const progressCounter = document.getElementById('progress-counter');
            const currentPaper = document.getElementById('current-paper');
            const progressTitle = document.getElementById('progress-title');
            const resultsList = document.getElementById('results-list');
            
            if (status.running) {{
                const percentage = status.total > 0 ? Math.round((status.progress / status.total) * 100) : 0;
                progressFill.style.width = percentage + '%';
                progressFill.textContent = percentage + '%';
                progressCounter.textContent = `${{status.progress}} / ${{status.total}}`;
                progressTitle.textContent = `🚀 ${{status.method.toUpperCase()}} Analysis in Progress...`;
                
                if (status.current_paper) {{
                    currentPaper.textContent = `Analyzing: ${{status.current_paper}}`;
                }}
                
                // Add new results to real-time display
                if (status.results && status.results.length > resultsList.children.length) {{
                    const newResults = status.results.slice(resultsList.children.length);
                    newResults.forEach(result => {{
                        const resultItem = document.createElement('div');
                        resultItem.className = `result-item ${{result.recommendation.toLowerCase()}}`;
                        resultItem.innerHTML = `
                            <div class="result-title">${{result.title}}</div>
                            <div class="result-score ${{result.recommendation.toLowerCase()}}">${{result.recommendation}} (${{result.overall_score.toFixed(2)}})</div>
                            <div class="result-time">${{result.analysis_time}}s</div>
                        `;
                        resultsList.appendChild(resultItem);
                        
                        // Scroll to bottom
                        resultsList.scrollTop = resultsList.scrollHeight;
                    }});
                }}
            }}
        }}
        
        function analysisCompleted() {{
            isAnalysisRunning = false;
            
            // Update UI
            document.getElementById('progress-section').classList.remove('active');
            document.getElementById('realtime-results').classList.remove('active');
            
            // Update buttons
            document.getElementById('regex-btn').disabled = false;
            document.getElementById('llm-btn').disabled = false;
            document.getElementById('stop-btn').style.display = 'none';
            document.getElementById('view-results-btn').style.display = 'inline-block';
            
            // Load and show final results
            loadResults();
        }}
        
        function resetUI() {{
            isAnalysisRunning = false;
            if (analysisInterval) {{
                clearInterval(analysisInterval);
                analysisInterval = null;
            }}
            
            document.getElementById('progress-section').classList.remove('active');
            document.getElementById('realtime-results').classList.remove('active');
            document.getElementById('regex-btn').disabled = false;
            document.getElementById('llm-btn').disabled = false;
            document.getElementById('stop-btn').style.display = 'none';
        }}
        
        function loadResults() {{
            fetch('/api/results')
                .then(response => response.json())
                .then(results => {{
                    displayResults(results);
                    document.getElementById('final-results').classList.add('active');
                }})
                .catch(error => {{
                    console.error('Error loading results:', error);
                }});
        }}
        
        function displayResults(results) {{
            if (results.regex) {{
                displayMethodResults('regex', results.regex);
            }}
            if (results.llm) {{
                displayMethodResults('llm', results.llm);
            }}
        }}
        
        function displayMethodResults(method, data) {{
            const container = document.getElementById(`${{method}}-results`);
            
            if (!data || !data.results || data.results.length === 0) {{
                container.innerHTML = '<div class="loading">No results available. Run analysis first.</div>';
                return;
            }}
            
            const summary = data.summary || {{}};
            const papers = data.results || [];
            
            // Generate stats
            const statsHtml = `
                <div class="stats">
                    <div class="stat-card total">
                        <div class="stat-number">${{summary.total || 0}}</div>
                        <div class="stat-label">Total</div>
                    </div>
                    <div class="stat-card include">
                        <div class="stat-number">${{summary.include || 0}}</div>
                        <div class="stat-label">Include</div>
                    </div>
                    <div class="stat-card exclude">
                        <div class="stat-number">${{summary.exclude || 0}}</div>
                        <div class="stat-label">Exclude</div>
                    </div>
                    <div class="stat-card review">
                        <div class="stat-number">${{summary.review || 0}}</div>
                        <div class="stat-label">Review</div>
                    </div>
                </div>
            `;
            
            // Generate papers grid
            const papersHtml = papers.map(paper => {{
                const criteriaHtml = Object.entries(paper.evaluations || {{}}).map(([name, eval]) => {{
                    const cssClass = eval.answer.toLowerCase() === 'yes' ? 'yes' : 
                                   eval.answer.toLowerCase() === 'no' ? 'no' : 'unknown';
                    return `<div class="criterion ${{cssClass}}">${{name}}: ${{eval.answer}}</div>`;
                }}).join('');
                
                return `
                    <div class="paper-card ${{paper.recommendation.toLowerCase()}}">
                        <div class="paper-title">${{paper.title}}</div>
                        <div class="paper-score">${{paper.recommendation}} (Score: ${{paper.overall_score.toFixed(2)}})</div>
                        <div class="criteria">${{criteriaHtml}}</div>
                    </div>
                `;
            }}).join('');
            
            container.innerHTML = statsHtml + '<div class="papers-grid">' + papersHtml + '</div>';
        }}
        
        function showResultsTab(method) {{
            // Update tab buttons
            document.querySelectorAll('.results-tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            event.target.classList.add('active');
            
            // Update content
            document.querySelectorAll('.results-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.getElementById(`${{method}}-results`).classList.add('active');
        }}
        
        // Load existing results on page load
        window.addEventListener('load', () => {{
            loadResults();
        }});
    </script>
</body>
</html>"""
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def main():
    """Start the real-time analysis web server"""
    port = 3444
    
    print(f"🚀 Starting Real-time Analysis UI on port {port}...")
    
    server = HTTPServer(('localhost', port), RealtimeAnalysisHandler)
    
    # Open browser
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open(f'http://localhost:{port}')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print(f"🌐 Real-time Analysis UI available at: http://localhost:{port}")
    print("🔬 Shows LIVE progress for each paper as it's analyzed!")
    print("📊 Real-time results with all 5 criteria!")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
