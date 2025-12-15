#!/usr/bin/env python3
"""
Minimal Web UI for Paper Analysis Pipeline
Uses only built-in Python modules + http.server
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

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.markdown_parser import MarkdownParser, DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer
from src.core.analyzer import PaperAnalyzer

# Try to import Bedrock evaluator
try:
    from src.evaluators.bedrock_evaluator import ParallelBedrockEvaluator
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    print("Bedrock evaluator not available")

# Try to import LLM evaluator (fallback)
try:
    from src.evaluators.llm_evaluator import LLMPaperEvaluator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class PaperAnalysisHandler(BaseHTTPRequestHandler):
    """HTTP request handler for paper analysis UI"""
    
    def __init__(self, *args, **kwargs):
        # Initialize components (shared across requests)
        if not hasattr(PaperAnalysisHandler, '_components_initialized'):
            self.setup_components()
            PaperAnalysisHandler._components_initialized = True
        super().__init__(*args, **kwargs)
    
    def setup_components(self):
        """Initialize analysis components"""
        print("🔧 Setting up analysis components...")
        
        db_path = Path("data/markdown_db")
        vector_path = Path("data/vector_store")
        
        PaperAnalysisHandler.parser = MarkdownParser()
        PaperAnalysisHandler.db = DocumentDatabase(db_path)
        PaperAnalysisHandler.vector_store = SimpleVectorStore(vector_path)
        PaperAnalysisHandler.criteria_analyzer = CriteriaAnalyzer(PaperAnalysisHandler.vector_store)
        PaperAnalysisHandler.regex_analyzer = PaperAnalyzer(PaperAnalysisHandler.db, PaperAnalysisHandler.vector_store)
        
        # Initialize Bedrock evaluator (parallel)
        if BEDROCK_AVAILABLE:
            PaperAnalysisHandler.bedrock_analyzer = ParallelBedrockEvaluator(
                PaperAnalysisHandler.db, 
                PaperAnalysisHandler.vector_store,
                max_workers=10  # Parallel requests
            )
        else:
            PaperAnalysisHandler.bedrock_analyzer = None
        
        # Fallback to OpenRouter LLM
        if LLM_AVAILABLE:
            PaperAnalysisHandler.llm_analyzer = LLMPaperEvaluator(PaperAnalysisHandler.db, PaperAnalysisHandler.vector_store)
        else:
            PaperAnalysisHandler.llm_analyzer = None
        
        # Analysis state
        PaperAnalysisHandler.analysis_running = False
        PaperAnalysisHandler.analysis_thread = None
        
        print("✓ Components initialized")
        print(f"  - Bedrock: {'✓ Available' if BEDROCK_AVAILABLE else '✗ Not available'}")
        print(f"  - LLM Fallback: {'✓ Available' if LLM_AVAILABLE else '✗ Not available'}")
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/papers':
            self.serve_papers_list()
        elif self.path == '/results':
            self.serve_results()
        elif self.path.startswith('/api/stats'):
            self.serve_api_stats()
        elif self.path.startswith('/api/analyze/'):
            parts = self.path.split('/')
            if len(parts) >= 5:
                method = parts[3]
                doc_id = parts[4]
                self.serve_api_analyze(method, doc_id)
        elif self.path == '/api/analysis-progress':
            self.serve_analysis_progress()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/analyze':
            self.handle_analyze_request()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Serve main dashboard with tabs and real-time analysis"""
        stats = self.get_system_stats()
        recent_papers = self.get_recent_papers()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Paper Analysis Pipeline</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; min-height: 100vh; }}
        .header {{ background: #343a40; color: white; padding: 20px; text-align: center; }}
        
        /* Tab Navigation */
        .tabs {{ display: flex; background: #dee2e6; border-bottom: 1px solid #ccc; }}
        .tab {{ padding: 15px 25px; cursor: pointer; border: none; background: none; font-size: 16px; }}
        .tab.active {{ background: white; border-bottom: 3px solid #007bff; }}
        .tab:hover {{ background: #f8f9fa; }}
        
        /* Tab Content */
        .tab-content {{ display: none; padding: 20px; }}
        .tab-content.active {{ display: block; }}
        
        /* Dashboard Styles */
        .stats {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ flex: 1; background: #007bff; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-card.success {{ background: #28a745; }}
        .stat-card.info {{ background: #17a2b8; }}
        .stat-card.warning {{ background: #ffc107; color: #333; }}
        
        .actions {{ display: flex; gap: 10px; margin-bottom: 30px; }}
        .btn {{ padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }}
        .btn-primary {{ background: #007bff; color: white; }}
        .btn-success {{ background: #28a745; color: white; }}
        .btn-info {{ background: #17a2b8; color: white; }}
        .btn-purple {{ background: #6f42c1; color: white; }}
        .btn-danger {{ background: #dc3545; color: white; }}
        .btn:hover {{ opacity: 0.8; }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        
        /* Analysis Results */
        .analysis-container {{ margin-top: 20px; }}
        .analysis-header {{ background: #f8f9fa; padding: 15px; border-radius: 8px 8px 0 0; border: 1px solid #dee2e6; }}
        .analysis-results {{ border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 8px 8px; max-height: 500px; overflow-y: auto; }}
        
        .paper-result {{ padding: 15px; border-bottom: 1px solid #eee; }}
        .paper-result.include {{ border-left: 4px solid #28a745; }}
        .paper-result.exclude {{ border-left: 4px solid #dc3545; }}
        .paper-result.review {{ border-left: 4px solid #ffc107; }}
        
        .paper-title {{ font-weight: bold; margin-bottom: 8px; }}
        .paper-score {{ font-size: 1.1em; margin-bottom: 10px; }}
        .criteria-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .criterion {{ padding: 4px 8px; border-radius: 4px; font-size: 0.85em; }}
        .criterion.yes {{ background: #d4edda; color: #155724; }}
        .criterion.no {{ background: #f8d7da; color: #721c24; }}
        .criterion.unknown {{ background: #e2e3e5; color: #383d41; }}
        
        /* Papers List */
        .papers-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
        .paper-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .paper-card-title {{ font-weight: bold; margin-bottom: 10px; }}
        .paper-meta {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
        .paper-actions {{ display: flex; gap: 8px; }}
        .btn-sm {{ padding: 6px 12px; font-size: 0.85em; }}
        
        /* Loading and Status */
        .loading {{ text-align: center; padding: 20px; color: #007bff; }}
        .status {{ margin: 10px 0; padding: 15px; border-radius: 4px; }}
        .status.success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
        .status.error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
        .status.info {{ background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }}
        
        /* Live Paper Status */
        .live-papers {{ margin-top: 20px; max-height: 600px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 8px; }}
        .live-paper-item {{ padding: 12px; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 15px; }}
        .live-paper-item.pending {{ background: #f8f9fa; }}
        .live-paper-item.running {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .live-paper-item.completed {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .live-paper-item.failed {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        
        .live-paper-status {{ min-width: 80px; font-weight: bold; font-size: 0.85em; }}
        .live-paper-title {{ flex: 1; font-weight: 500; }}
        .live-paper-progress {{ min-width: 60px; text-align: right; font-size: 0.85em; }}
        .live-criteria-tags {{ display: flex; gap: 4px; flex-wrap: wrap; }}
        .live-tag {{ padding: 2px 6px; border-radius: 3px; font-size: 0.75em; font-weight: bold; cursor: help; position: relative; }}
        .live-tag.yes {{ background: #28a745; color: white; }}
        .live-tag.no {{ background: #dc3545; color: white; }}
        .live-tag.unknown {{ background: #6c757d; color: white; }}
        .live-tag.pending {{ background: #e9ecef; color: #6c757d; }}
        
        /* Tooltip styles */
        .live-tag[title]:hover::after {{
            content: attr(title);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            white-space: normal;
            width: 300px;
            font-size: 0.9em;
            font-weight: normal;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            line-height: 1.4;
        }}
        .live-tag[title]:hover::before {{
            content: '';
            position: absolute;
            bottom: 115%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: #333;
            z-index: 1000;
        }}
        
        .live-recommendation {{ min-width: 80px; text-align: center; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }}
        .live-recommendation.Include {{ background: #28a745; color: white; }}
        .live-recommendation.Exclude {{ background: #dc3545; color: white; }}
        .live-recommendation.Review {{ background: #ffc107; color: #333; }}
        .live-recommendation.Pending {{ background: #e9ecef; color: #6c757d; }}
        
        .live-score {{ min-width: 60px; text-align: right; font-weight: bold; color: #007bff; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 Paper Analysis Pipeline</h1>
            <p>Real-time Analysis with Deep Researcher Criteria</p>
        </div>
        
        <!-- Tab Navigation -->
        <div class="tabs">
            <button class="tab active" onclick="showTab('dashboard')">🏠 Dashboard</button>
            <button class="tab" onclick="showTab('papers')">📋 Papers</button>
        </div>
        
        <!-- Dashboard Tab -->
        <div id="dashboard" class="tab-content active">
            <div class="stats">
                <div class="stat-card">
                    <h3>{stats['total_documents']}</h3>
                    <p>Documents</p>
                </div>
                <div class="stat-card success">
                    <h3>{stats['vector_embeddings']}</h3>
                    <p>Embeddings</p>
                </div>
                <div class="stat-card info">
                    <h3>{stats['criteria_count']}</h3>
                    <p>Criteria</p>
                </div>
                <div class="stat-card warning">
                    <h3>{'Yes' if stats['llm_available'] else 'No'}</h3>
                    <p>LLM Available</p>
                </div>
            </div>
            
            <div class="actions">
                <button id="analyze-regex-btn" class="btn btn-success" onclick="startAnalysis('regex')">
                    🔍 Analyze All (Regex)
                </button>
                {'<button id="analyze-bedrock-btn" class="btn btn-primary" onclick="startAnalysis(\'bedrock\')" title="Runs 1 mega-prompt per paper (all 5 criteria at once)">🚀 Analyze All (Bedrock - PARALLEL)</button>' if stats['bedrock_available'] else ''}
                {'<button id="analyze-bedrock-separate-btn" class="btn btn-purple" onclick="startAnalysis(\'bedrock-separate\')" title="Runs 5 separate API calls per paper (one for each criterion)">🔬 Analyze All (Bedrock - SEPARATE)</button>' if stats['bedrock_available'] else ''}
                {'<button id="analyze-llm-btn" class="btn btn-info" onclick="startAnalysis(\'llm\')">🧠 Analyze All (LLM)</button>' if stats['llm_available'] else ''}
                <button class="btn btn-danger" onclick="stopAnalysis()" style="display: none;" id="stop-btn">
                    ⏹️ Stop Analysis
                </button>
            </div>
            
            <!-- Analysis Status and Results -->
            <div id="analysis-status" style="display: none;"></div>
            
            <!-- Live Paper Status List -->
            <div id="live-papers-container" style="display: none;">
                <h3>📋 Live Analysis Progress</h3>
                <div id="live-summary" style="margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px;"></div>
                <div id="live-papers" class="live-papers"></div>
            </div>
            
            <div id="analysis-container" class="analysis-container" style="display: none;">
                <div class="analysis-header">
                    <h3 id="analysis-title">Analysis Results</h3>
                    <div id="analysis-summary"></div>
                </div>
                <div id="analysis-results" class="analysis-results"></div>
            </div>
        </div>
        
        <!-- Papers Tab -->
        <div id="papers" class="tab-content">
            <h3>📚 All Papers ({len(self.db.list_documents())} total)</h3>
            <div class="papers-grid">
                {self.render_papers_grid()}
            </div>
        </div>
    </div>
    
    <script>
        let analysisInProgress = false;
        let currentMethod = '';
        let pollInterval = null;
        
        const CRITERIA_NAMES = {{
            'pytorch': 'PyTorch',
            'supervised': 'Supervised',
            'small_dataset': 'Small DS',
            'quick_training': 'Quick Train',
            'has_repo': 'Has Repo'
        }};
        
        function showTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
        
        function startAnalysis(method) {{
            if (analysisInProgress) return;
            
            analysisInProgress = true;
            currentMethod = method;
            
            // Update UI
            document.getElementById('analyze-regex-btn').disabled = true;
            if (document.getElementById('analyze-bedrock-btn'))
                document.getElementById('analyze-bedrock-btn').disabled = true;
            if (document.getElementById('analyze-bedrock-separate-btn'))
                document.getElementById('analyze-bedrock-separate-btn').disabled = true;
            if (document.getElementById('analyze-llm-btn'))
                document.getElementById('analyze-llm-btn').disabled = true;
            document.getElementById('stop-btn').style.display = 'inline-block';
            
            // Show status
            const statusDiv = document.getElementById('analysis-status');
            statusDiv.className = 'status info';
            statusDiv.innerHTML = `🔄 Starting ${{method.toUpperCase()}} analysis...`;
            statusDiv.style.display = 'block';
            
            // Show live papers container
            document.getElementById('live-papers-container').style.display = 'block';
            document.getElementById('analysis-container').style.display = 'none';
            
            // Start the analysis (non-blocking)
            fetch('/analyze', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
                body: 'method=' + method
            }})
            .then(response => response.json())
            .then(data => {{
                console.log('Analysis started:', data);
                // Start polling for progress
                pollInterval = setInterval(pollResults, 500);  // Poll every 500ms
            }})
            .catch(error => {{
                console.error('Error starting analysis:', error);
                finishAnalysis({{error: error.toString()}});
            }});
        }}
        
        function pollResults() {{
            fetch('/api/analysis-progress')
            .then(response => response.json())
            .then(data => {{
                updateLivePapers(data);
                
                // Check if analysis is done
                if (!data.running && data.final_results) {{
                    clearInterval(pollInterval);
                    finishAnalysis(data.final_results);
                }}
            }})
            .catch(error => {{
                console.error('Polling error:', error);
            }});
        }}
        
        function updateLivePapers(data) {{
            const livePapers = document.getElementById('live-papers');
            const liveSummary = document.getElementById('live-summary');
            
            if (!data.papers || data.papers.length === 0) return;
            
            // Update summary
            const completed = data.completed || 0;
            const total = data.total || 0;
            const pending = data.papers.filter(p => p.status === 'pending').length;
            const running = data.papers.filter(p => p.status === 'running').length;
            const failed = data.papers.filter(p => p.status === 'failed').length;
            
            liveSummary.innerHTML = `
                <strong>Progress:</strong> ${{completed}}/${{total}} completed 
                | ⏳ Pending: ${{pending}} 
                | 🔄 Running: ${{running}} 
                | ❌ Failed: ${{failed}}
            `;
            
            // Update paper list
            livePapers.innerHTML = '';
            
            data.papers.forEach(paper => {{
                const div = document.createElement('div');
                div.className = `live-paper-item ${{paper.status}}`;
                
                // Status icon
                let statusIcon = '';
                switch(paper.status) {{
                    case 'pending': statusIcon = '⏳'; break;
                    case 'running': statusIcon = '🔄'; break;
                    case 'completed': statusIcon = '✅'; break;
                    case 'failed': statusIcon = '❌'; break;
                }}
                
                // Criteria tags
                const criteriaHtml = Object.keys(CRITERIA_NAMES).map(crit => {{
                    const answer = (paper.criteria && paper.criteria[crit]) || 'pending';
                    const className = answer.toLowerCase();
                    
                    // Get evidence for tooltip
                    let evidence = '';
                    let confidence = '';
                    if (paper.criteria_evidence && paper.criteria_evidence[crit]) {{
                        evidence = paper.criteria_evidence[crit].evidence || 'No evidence provided';
                        confidence = paper.criteria_evidence[crit].confidence || '';
                        const confidenceStr = confidence ? ` (confidence: ${{(confidence * 100).toFixed(0)}}%)` : '';
                        const newline = '\\n';
                        const tooltip = `${{CRITERIA_NAMES[crit]}}: ${{answer}}${{confidenceStr}}${{newline}}${{newline}}Evidence: ${{evidence}}`;
                        return `<span class="live-tag ${{className}}" title="${{tooltip.replace(/"/g, '&quot;')}}">${{CRITERIA_NAMES[crit]}}</span>`;
                    }} else {{
                        return `<span class="live-tag ${{className}}">${{CRITERIA_NAMES[crit]}}</span>`;
                    }}
                }}).join('');
                
                // Recommendation
                const rec = paper.recommendation || 'Pending';
                const recClass = rec.replace(' ', '-');
                
                div.innerHTML = `
                    <div class="live-paper-status">${{statusIcon}} ${{paper.status}}</div>
                    <div class="live-paper-title">${{paper.title.substring(0, 60)}}${{paper.title.length > 60 ? '...' : ''}}</div>
                    <div class="live-criteria-tags">${{criteriaHtml}}</div>
                    <div class="live-recommendation ${{recClass}}">${{rec}}</div>
                    <div class="live-score">${{paper.score ? paper.score.toFixed(2) : '—'}}</div>
                `;
                
                livePapers.appendChild(div);
            }});
        }}
        
        function finishAnalysis(data) {{
            analysisInProgress = false;
            if (pollInterval) {{
                clearInterval(pollInterval);
                pollInterval = null;
            }}
            
            // Update UI
            document.getElementById('analyze-regex-btn').disabled = false;
            if (document.getElementById('analyze-bedrock-btn'))
                document.getElementById('analyze-bedrock-btn').disabled = false;
            if (document.getElementById('analyze-bedrock-separate-btn'))
                document.getElementById('analyze-bedrock-separate-btn').disabled = false;
            if (document.getElementById('analyze-llm-btn'))
                document.getElementById('analyze-llm-btn').disabled = false;
            document.getElementById('stop-btn').style.display = 'none';
            
            // Update status
            const statusDiv = document.getElementById('analysis-status');
            if (data.error) {{
                statusDiv.className = 'status error';
                statusDiv.innerHTML = `❌ Analysis failed: ${{data.error}}`;
            }} else {{
                statusDiv.className = 'status success';
                statusDiv.innerHTML = `✅ Analysis complete! ${{data.total_papers || 0}} papers analyzed | 
                    Include: ${{data.include_count || 0}} | 
                    Exclude: ${{data.exclude_count || 0}} | 
                    Review: ${{data.review_count || 0}}`;
            }}
        }}
        
        function stopAnalysis() {{
            analysisInProgress = false;
            if (pollInterval) {{
                clearInterval(pollInterval);
                pollInterval = null;
            }}
            document.getElementById('stop-btn').style.display = 'none';
            document.getElementById('analyze-regex-btn').disabled = false;
            if (document.getElementById('analyze-bedrock-btn'))
                document.getElementById('analyze-bedrock-btn').disabled = false;
            if (document.getElementById('analyze-bedrock-separate-btn'))
                document.getElementById('analyze-bedrock-separate-btn').disabled = false;
            if (document.getElementById('analyze-llm-btn'))
                document.getElementById('analyze-llm-btn').disabled = false;
            
            const statusDiv = document.getElementById('analysis-status');
            statusDiv.className = 'status info';
            statusDiv.innerHTML = '⏹️ Analysis stopped by user';
        }}
        
        function analyzeSingle(docId, method) {{
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            fetch(`/api/analyze/${{method}}/${{docId}}`)
            .then(response => response.json())
            .then(data => {{
                if (data.error) {{
                    alert('Error: ' + data.error);
                }} else {{
                    alert(`Analysis Complete!\nScore: ${{data.overall_score.toFixed(2)}}\nRecommendation: ${{data.recommendation}}`);
                }}
                btn.disabled = false;
                btn.textContent = `Analyze (${{method.toUpperCase()}})`;
            }})
            .catch(error => {{
                alert('Analysis failed: ' + error);
                btn.disabled = false;
                btn.textContent = `Analyze (${{method.toUpperCase()}})`;
            }});
        }}
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_papers_list(self):
        """Serve papers list page"""
        papers = self.db.list_documents()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Papers List - Paper Analysis Pipeline</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .nav {{ margin-bottom: 20px; }}
        .nav a {{ margin-right: 15px; text-decoration: none; color: #007bff; }}
        .paper-card {{ background: #f8f9fa; margin: 15px 0; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .paper-title {{ font-weight: bold; font-size: 1.1em; color: #333; margin-bottom: 10px; }}
        .paper-meta {{ color: #666; font-size: 0.9em; }}
        .analyze-btn {{ background: #28a745; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; margin: 5px; }}
        .analyze-btn:hover {{ opacity: 0.8; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📋 Papers List</h1>
        </div>
        
        <div class="nav">
            <a href="/">🏠 Dashboard</a>
            <a href="/papers">📋 Papers</a>
            <a href="/results">📊 Results</a>
        </div>
        
        <p><strong>Total Papers:</strong> {len(papers)}</p>
        
        <div class="papers-container">
            {self.render_detailed_papers_list(papers)}
        </div>
    </div>
    
    <script>
        function analyzeSingle(docId, method) {{
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            fetch(`/api/analyze/${{method}}/${{docId}}`)
            .then(response => response.json())
            .then(data => {{
                if (data.error) {{
                    alert('Error: ' + data.error);
                }} else {{
                    alert(`Analysis Complete!\nScore: ${{data.overall_score.toFixed(2)}}\nRecommendation: ${{data.recommendation}}`);
                }}
                btn.disabled = false;
                btn.textContent = `Analyze (${{method.toUpperCase()}})`;
            }})
            .catch(error => {{
                alert('Analysis failed: ' + error);
                btn.disabled = false;
                btn.textContent = `Analyze (${{method.toUpperCase()}})`;
            }});
        }}
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_results(self):
        """Serve results page"""
        # Find latest results file
        result_files = list(Path("archive/old_analysis_results").glob("*_analysis_*.json"))
        
        if result_files:
            latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file, 'r') as f:
                results_data = json.load(f)
        else:
            results_data = {"results": [], "summary": {}}
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Results - Paper Analysis Pipeline</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .nav {{ margin-bottom: 20px; }}
        .nav a {{ margin-right: 15px; text-decoration: none; color: #007bff; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ flex: 1; padding: 15px; border-radius: 8px; text-align: center; color: white; }}
        .summary-card.include {{ background: #28a745; }}
        .summary-card.exclude {{ background: #dc3545; }}
        .summary-card.review {{ background: #ffc107; color: #333; }}
        .summary-card.total {{ background: #007bff; }}
        .result-item {{ background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 8px; }}
        .result-item.include {{ border-left: 4px solid #28a745; }}
        .result-item.exclude {{ border-left: 4px solid #dc3545; }}
        .result-item.review {{ border-left: 4px solid #ffc107; }}
        .result-title {{ font-weight: bold; margin-bottom: 10px; }}
        .result-score {{ font-size: 1.2em; color: #007bff; }}
        .criteria {{ display: flex; gap: 10px; margin-top: 10px; }}
        .criterion {{ padding: 3px 8px; border-radius: 4px; font-size: 0.8em; }}
        .criterion.yes {{ background: #d4edda; color: #155724; }}
        .criterion.no {{ background: #f8d7da; color: #721c24; }}
        .criterion.unknown {{ background: #e2e3e5; color: #383d41; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Analysis Results</h1>
        </div>
        
        <div class="nav">
            <a href="/">🏠 Dashboard</a>
            <a href="/papers">📋 Papers</a>
            <a href="/results">📊 Results</a>
        </div>
        
        {self.render_results_summary(results_data)}
        
        <div class="results-container">
            {self.render_results_list(results_data.get('results', []))}
        </div>
    </div>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_api_stats(self):
        """Serve API stats endpoint"""
        stats = self.get_system_stats()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())
    
    def serve_api_analyze(self, method, doc_id):
        """Serve API analyze endpoint"""
        try:
            if method == 'llm' and not self.llm_analyzer:
                result = {"error": "LLM analyzer not available"}
            else:
                analyzer = self.llm_analyzer if method == 'llm' else self.regex_analyzer
                analysis_result = analyzer.analyze_paper(doc_id)
                
                if analysis_result:
                    result = {
                        "doc_id": analysis_result.doc_id,
                        "title": analysis_result.title,
                        "overall_score": analysis_result.overall_score,
                        "recommendation": analysis_result.recommendation,
                        "evaluations": {
                            name: {
                                "answer": eval.answer,
                                "confidence": eval.confidence,
                                "evidence": eval.evidence
                            }
                            for name, eval in analysis_result.evaluations.items()
                        }
                    }
                else:
                    result = {"error": "Analysis failed"}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def handle_analyze_request(self):
        """Handle analyze POST request with parallel execution"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            params = urllib.parse.parse_qs(post_data)
            
            method = params.get('method', ['regex'])[0]
            
            # Check if already running
            if PaperAnalysisHandler.analysis_running:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Analysis already running"}).encode())
                return
            
            # Start analysis in background thread
            PaperAnalysisHandler.analysis_running = True
            
            def run_analysis():
                try:
                    if method == 'bedrock' and self.bedrock_analyzer:
                        print("🚀 Starting PARALLEL Bedrock analysis (1 call per paper)...")
                        results = self.bedrock_analyzer.analyze_all_papers_parallel()
                        
                        # Store results for tooltips
                        if not hasattr(PaperAnalysisHandler, '_paper_results'):
                            PaperAnalysisHandler._paper_results = {}
                        for result in results:
                            PaperAnalysisHandler._paper_results[result.doc_id] = result
                        
                        # Save results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = Path(f"bedrock_parallel_analysis_{timestamp}.json")
                        self.bedrock_analyzer.save_results(results, output_file)
                        
                        PaperAnalysisHandler._analysis_results = {
                            "method": "bedrock",
                            "total_papers": len(results),
                            "include_count": sum(1 for r in results if r.recommendation == "Include"),
                            "exclude_count": sum(1 for r in results if r.recommendation == "Exclude"),
                            "review_count": sum(1 for r in results if r.recommendation == "Review"),
                            "output_file": str(output_file)
                        }
                    
                    elif method == 'bedrock-separate' and self.bedrock_analyzer:
                        print("🔬 Starting SEPARATE criteria Bedrock analysis (5 calls per paper)...")
                        results = self.bedrock_analyzer.analyze_all_papers_separate_criteria()
                        
                        # Store results for tooltips
                        if not hasattr(PaperAnalysisHandler, '_paper_results'):
                            PaperAnalysisHandler._paper_results = {}
                        for result in results:
                            PaperAnalysisHandler._paper_results[result.doc_id] = result
                        
                        # Save results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = Path(f"bedrock_separate_analysis_{timestamp}.json")
                        self.bedrock_analyzer.save_results(results, output_file)
                        
                        PaperAnalysisHandler._analysis_results = {
                            "method": "bedrock-separate",
                            "total_papers": len(results),
                            "include_count": sum(1 for r in results if r.recommendation == "Include"),
                            "exclude_count": sum(1 for r in results if r.recommendation == "Exclude"),
                            "review_count": sum(1 for r in results if r.recommendation == "Review"),
                            "output_file": str(output_file)
                        }
                        
                    elif method == 'llm' and self.llm_analyzer:
                        results = self.llm_analyzer.analyze_all_papers()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = Path(f"llm_analysis_{timestamp}.json")
                        self.llm_analyzer.save_results(results, output_file)
                        
                        PaperAnalysisHandler._analysis_results = {
                            "method": "llm",
                            "total_papers": len(results),
                            "output_file": str(output_file)
                        }
                        
                    else:  # regex
                        papers = self.db.list_documents()
                        results = []
                        for paper_info in papers:
                            result = self.regex_analyzer.analyze_paper(paper_info['doc_id'])
                            if result:
                                results.append(result)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = Path(f"regex_analysis_{timestamp}.json")
                        self.regex_analyzer.save_results(results, output_file)
                        
                        PaperAnalysisHandler._analysis_results = {
                            "method": "regex",
                            "total_papers": len(results),
                            "output_file": str(output_file)
                        }
                    
                except Exception as e:
                    print(f"❌ Analysis error: {e}")
                    import traceback
                    traceback.print_exc()
                    PaperAnalysisHandler._analysis_results = {"error": str(e)}
                finally:
                    PaperAnalysisHandler.analysis_running = False
            
            # Start in background
            PaperAnalysisHandler.analysis_thread = threading.Thread(target=run_analysis, daemon=True)
            PaperAnalysisHandler.analysis_thread.start()
            
            # Return immediately
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "method": method}).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def get_system_stats(self):
        """Get system statistics"""
        return {
            "total_documents": len(self.db.documents),
            "vector_embeddings": self.vector_store.get_stats()["total_embeddings"],
            "criteria_count": len(self.regex_analyzer.criteria),
            "llm_available": LLM_AVAILABLE,
            "bedrock_available": BEDROCK_AVAILABLE,
        }
    
    def get_recent_papers(self, limit=5):
        """Get recent papers"""
        papers = self.db.list_documents()
        return sorted(papers, key=lambda p: p.get('parsed_at', ''), reverse=True)[:limit]
    
    def render_papers_list(self, papers):
        """Render papers list HTML"""
        if not papers:
            return "<p>No papers found.</p>"
        
        html = ""
        for paper in papers:
            html += f"""
            <div class="paper-item">
                <div class="paper-title">{paper['title']}</div>
                <div class="paper-meta">
                    Sections: {paper['section_count']} | 
                    ID: {paper['doc_id'][:8]}... | 
                    File: {paper['file_path']}
                </div>
            </div>
            """
        return html
    
    def render_detailed_papers_list(self, papers):
        """Render detailed papers list with analyze buttons"""
        if not papers:
            return "<p>No papers found.</p>"
        
        html = ""
        for paper in papers:
            html += f"""
            <div class="paper-card">
                <div class="paper-title">{paper['title']}</div>
                <div class="paper-meta">
                    Sections: {paper['section_count']} | 
                    ID: {paper['doc_id'][:8]}... | 
                    File: {paper['file_path']}
                </div>
                <div style="margin-top: 10px;">
                    <button class="analyze-btn" onclick="analyzeSingle('{paper['doc_id']}', 'regex')">Analyze (Regex)</button>
                    {'<button class="analyze-btn" onclick="analyzeSingle(\'' + paper['doc_id'] + '\', \'llm\')">Analyze (LLM)</button>' if LLM_AVAILABLE else ''}
                </div>
            </div>
            """
        return html
    
    def render_results_summary(self, results_data):
        """Render results summary"""
        summary = results_data.get('summary', {})
        
        return f"""
        <div class="summary">
            <div class="summary-card total">
                <h3>{summary.get('total', 0)}</h3>
                <p>Total Papers</p>
            </div>
            <div class="summary-card include">
                <h3>{summary.get('include', 0)}</h3>
                <p>Include</p>
            </div>
            <div class="summary-card exclude">
                <h3>{summary.get('exclude', 0)}</h3>
                <p>Exclude</p>
            </div>
            <div class="summary-card review">
                <h3>{summary.get('review', 0)}</h3>
                <p>Review</p>
            </div>
        </div>
        """
    
    def render_results_list(self, results):
        """Render results list"""
        if not results:
            return "<p>No results found. Run an analysis first.</p>"
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x.get('overall_score', 0), reverse=True)
        
        html = ""
        for result in sorted_results[:20]:  # Show top 20
            recommendation = result.get('recommendation', 'Unknown').lower()
            
            # Render criteria
            criteria_html = ""
            evaluations = result.get('evaluations', {})
            for criterion_name, evaluation in evaluations.items():
                answer = evaluation.get('answer', 'Unknown').lower()
                criteria_html += f'<span class="criterion {answer}">{criterion_name}: {evaluation.get("answer", "Unknown")}</span>'
            
            html += f"""
            <div class="result-item {recommendation}">
                <div class="result-title">{result.get('title', 'Unknown')}</div>
                <div class="result-score">Score: {result.get('overall_score', 0):.2f} | {result.get('recommendation', 'Unknown')}</div>
                <div class="criteria">{criteria_html}</div>
            </div>
            """
        
        return html
    
    def render_papers_grid(self):
        """Render papers in grid layout"""
        papers = self.db.list_documents()
        if not papers:
            return "<p>No papers found.</p>"
        
        html = ""
        for paper in papers:
            html += f"""
            <div class="paper-card">
                <div class="paper-card-title">{paper['title']}</div>
                <div class="paper-meta">
                    Sections: {paper['section_count']} | 
                    ID: {paper['doc_id'][:8]}... | 
                    File: {Path(paper['file_path']).name}
                </div>
                <div class="paper-actions">
                    <button class="btn btn-sm btn-success" onclick="analyzeSingle('{paper['doc_id']}', 'regex')">
                        Analyze (Regex)
                    </button>
                    {'<button class="btn btn-sm btn-info" onclick="analyzeSingle(\'' + paper['doc_id'] + '\', \'llm\')">Analyze (LLM)</button>' if LLM_AVAILABLE else ''}
                </div>
            </div>
            """
        return html
    
    def serve_analysis_progress(self):
        """Serve analysis progress for real-time updates"""
        if self.bedrock_analyzer and hasattr(self, 'bedrock_analyzer'):
            # Get real-time statuses from Bedrock analyzer
            statuses = self.bedrock_analyzer.get_all_statuses()
            
            progress_data = {
                "running": PaperAnalysisHandler.analysis_running,
                "papers": [
                    {
                        "doc_id": status.doc_id,
                        "title": status.title,
                        "status": status.status,
                        "progress": status.progress,
                        "criteria": status.criteria_results,
                        "criteria_evidence": self._get_criteria_evidence(status.doc_id),
                        "score": status.overall_score,
                        "recommendation": status.recommendation,
                        "error": status.error
                    }
                    for status in statuses
                ],
                "completed": sum(1 for s in statuses if s.status == "completed"),
                "total": len(statuses),
                "final_results": getattr(PaperAnalysisHandler, '_analysis_results', None)
            }
        else:
            # Fallback for non-Bedrock analyzers
            progress_data = {
                "running": PaperAnalysisHandler.analysis_running,
                "papers": [],
                "completed": 0,
                "total": len(self.db.documents),
                "final_results": getattr(PaperAnalysisHandler, '_analysis_results', None)
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(progress_data).encode())
    
    def _get_criteria_evidence(self, doc_id):
        """Get evidence for each criterion from completed analysis"""
        if not hasattr(PaperAnalysisHandler, '_paper_results'):
            PaperAnalysisHandler._paper_results = {}
        
        if doc_id in PaperAnalysisHandler._paper_results:
            result = PaperAnalysisHandler._paper_results[doc_id]
            return {
                name: {
                    "evidence": eval.evidence[:150],  # Truncate for tooltip
                    "confidence": eval.confidence
                }
                for name, eval in result.evaluations.items()
            }
        return {}
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass


def run_server(port=3444):
    """Run the web server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, PaperAnalysisHandler)
    
    print(f"🚀 Starting Paper Analysis Web UI...")
    print(f"🌐 Server running at: http://localhost:{port}")
    print(f"📊 Access the dashboard in your browser")
    print(f"🔧 Press Ctrl+C to stop the server")
    
    # Try to open browser
    try:
        webbrowser.open(f'http://localhost:{port}')
    except:
        pass
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n👋 Shutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    run_server(3444)
