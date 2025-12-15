# Search and Classify Workflow

A workflow for finding and classifying relevant papers using multiple search sources and AWS Bedrock with Deepseek:
1. **Search**: Find papers using Semantic Scholar or ASTA Corpus
2. **Classify**: Filter papers using AWS Bedrock with Deepseek model
3. **Download**: Download PDFs from arXiv for relevant papers
4. **Visualize**: View classified papers in a table format

## Prerequisites

1. **Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **AWS Configuration**:
   ```bash
   # Configure AWS credentials (choose one method)
   aws configure  # Interactive setup
   
   # OR set environment variables
   export AWS_ACCESS_KEY_ID="your_access_key"
   export AWS_SECRET_ACCESS_KEY="your_secret_key"
   export AWS_REGION="us-east-1"
   ```

3. **Environment variables** (optional):
   ```bash
   export SEMANTIC_SCHOLAR_API_KEY="your_key"  # For Semantic Scholar (optional)
   export ASTA_API_KEY="your_key"              # For ASTA Corpus
   ```

## Configuration

All settings are configured via `config.yaml` using Hydra. Key sections:

- **ASTA**: Search queries and API settings for ASTA Corpus
- **Search**: Semantic Scholar API settings
- **Bedrock**: AWS Bedrock configuration with Deepseek model
- **Classification**: Research description and output settings
- **Download**: PDF download settings

You can override any config value using Hydra syntax:
```bash
python semantic_scholar.py search.query="new query" search.limit=500
```

## Workflow

### Step 1: Search Papers

You can search using either Semantic Scholar or ASTA Corpus:

#### Option A: Search with ASTA Corpus

```bash
python asta.py
```

**Configuration** (in `config.yaml`):
- `asta.search_queries`: List of search queries
- `asta.limit`: Papers per query
- `asta.output`: Output JSON file

#### Option B: Search with Semantic Scholar

```bash
python semantic_scholar.py
```

**Configuration** (in `config.yaml`):
- `search.query`: Search query
- `search.limit`: Maximum number of papers
- `search.output`: Output JSON file
- `search.show_abstract`: Display abstracts

**Output**: `search_results.json` (or configured output file) with all papers found

### Step 2: Classify Papers

Filter papers using AWS Bedrock with Deepseek model:

```bash
python classify_papers.py
```

**Configuration** (in `config.yaml`):
- `classification.research_description`: Detailed criteria for relevance
- `classification.output_prefix`: Output file prefix (default: "classified")
- `classification.display_results`: Show results summary (default: true)
- `classification.display_limit`: Number of papers to display (default: 10)
- `bedrock.region`: AWS region (default: "us-east-1")
- `bedrock.model_id`: Bedrock model ID for Deepseek
- `bedrock.max_tokens`: Maximum tokens in response (default: 10)
- `bedrock.temperature`: Temperature for generation (default: 0.1)

**Output**:
- `classified_papers.json` - All papers with classification labels:
  - `relevant`: Boolean indicating if paper matches criteria

### Step 3: Download PDFs (Optional)

Download PDFs from arXiv for papers classified as relevant:

```bash
python download_papers.py
```

**Configuration** (in `config.yaml`):
- `download.output_dir`: Directory to save PDFs (default: "downloaded_papers")
- `download.delay`: Delay between downloads in seconds (default: 1.0)

**Behavior**:
- Downloads papers classified as relevant
- Searches arXiv by title if URL not found in paper metadata
- Skips papers not available on arXiv

### Step 4: Visualize Results (Optional)

View classified papers in a table format:

```bash
python visualize_papers.py classified_papers.json
```

Shows papers grouped by classification:
- Both models: YES
- Models disagree
- Both models: NO

## Complete Example

### Using ASTA Corpus (Recommended)

```bash
# Step 1: Search for papers (uses config.yaml)
python asta.py

# Step 2: Classify papers (uses config.yaml)
python classify_papers.py

# Step 3: Download PDFs (optional)
python download_papers.py

# Step 4: Visualize results (optional)
python visualize_papers.py classified_papers.json
```

### Using Semantic Scholar

```bash
# Step 1: Search for papers (uses config.yaml)
python semantic_scholar.py

# Step 2: Classify papers (uses config.yaml)
python classify_papers.py

# Step 3: Download PDFs (optional)
python download_papers.py
```

### Overriding Configuration

You can override any config value using Hydra syntax:

```bash
# Override search query and limit
python semantic_scholar.py search.query="neural combinatorial optimization" search.limit=200

# Override research description
python classify_papers.py classification.research_description="Papers on neural optimization"

# Override classifier selection
python classify_papers.py classification.use_openrouter=false  # Use only VLLM
```

**Results**:
- `classified_papers.json` - All papers with `vllm_relevant` and `openrouter_relevant` labels
- `downloaded_papers/` - PDF files for relevant papers (if download step was run)

## Research Description Tips

The research description (configured in `config.yaml` under `classification.research_description`) is crucial for good classification. Be specific:

### Good Examples

✅ **Specific and detailed**:
```yaml
classification:
  research_description: |
    Papers that apply graph neural networks to Boolean satisfiability problems,
    including SAT solving, MAXSAT, or related constraint satisfaction problems.
    Papers should focus on using GNNs for solution prediction, search guidance,
    or formula representation.
```

✅ **Multiple criteria**:
```yaml
classification:
  research_description: |
    Papers on neural theorem proving OR automated reasoning with machine learning.
    Include papers on:
    - Proof search with neural networks
    - Premise selection using ML
    - Tactic prediction in proof assistants
    - Neural symbolic reasoning
```

### Less Effective

❌ **Too vague**:
```yaml
classification:
  research_description: "Papers on machine learning"
```

❌ **Too narrow**:
```yaml
classification:
  research_description: "Papers on GNNs for SAT published in 2023 by researchers at MIT"
```

## Output Format

### Classified Papers JSON

The output file contains all papers with classification labels:

```json
[
  {
    "paperId": "abc123",
    "title": "Graph Neural Networks for SAT Solving",
    "abstract": "We present a novel approach...",
    "year": 2023,
    "citationCount": 42,
    "authors": [...],
    "relevant": true,
    ...
  },
  {
    "paperId": "def456",
    "title": "Image Classification with Deep Learning",
    ...
    "relevant": false,
    ...
  }
]
```

### Terminal Output

```
================================================================================
CLASSIFYING 200 PAPERS WITH: AWS Bedrock (Deepseek)
================================================================================

[1/200] Graph Neural Networks for SAT Solving...
  Bedrock: ✓ YES

[2/200] Image Classification with Deep Learning...
  Bedrock: ✗ NO

[3/200] NeuroSAT: Learning a SAT Solver...
  Bedrock: ✓ YES

...

================================================================================
CLASSIFICATION COMPLETE
================================================================================
Total papers: 200
Relevant: 42 (21.0%)
================================================================================
```

## Performance

### Search Speed
- **ASTA**: ~5-10 seconds per query (depends on API response time)
- **Semantic Scholar**:
  - 100 papers: Instant (1 API request)
  - 500 papers: ~10 seconds (5 requests × 2s delay)
  - 1000 papers: ~20 seconds (10 requests × 2s delay)

### Classification Speed
Using AWS Bedrock:
- **Per paper**: ~1-3 seconds (depends on network latency)
- **100 papers**: ~3-5 minutes
- **500 papers**: ~15-25 minutes
- **1000 papers**: ~30-50 minutes

## Troubleshooting

### Search Issues

**Problem**: Rate limiting (HTTP 429)
**Solution**: Increase delay in `config.yaml`:
```yaml
search:
  delay: 5.0  # Increase from default 2.0
```

**Problem**: Not enough results
**Solution**:
- Use broader search query
- Increase `search.limit` in config
- Try different keywords
- Use multiple queries with ASTA (`asta.search_queries`)

**Problem**: ASTA API key error
**Solution**: Set environment variable:
```bash
export ASTA_API_KEY="your_key"
```

### Classification Issues

**Problem**: AWS Bedrock connection error
**Solution**:
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Reconfigure if needed
aws configure
```

**Problem**: Model access denied
**Solution**:
- Go to AWS Console → Bedrock → Model access
- Request access to the Deepseek/desired model
- Wait for approval (usually instant)

**Problem**: Invalid model ID
**Solution**:
```bash
# List available models in your region
aws bedrock list-foundation-models --region us-east-1

# Update config.yaml with valid model ID
```

**Problem**: All papers classified as irrelevant
**Solution**:
- Make research description more inclusive in `config.yaml`
- Check if papers have abstracts
- Try broader criteria
- Test with a few papers manually first

## Advanced Usage

### Using OpenRouter Instead of Bedrock

If you prefer OpenRouter:

1. Modify `classify_papers.py`:
```python
from classifiers import OpenRouterClassifier

# In classify_papers function:
classifier = OpenRouterClassifier(cfg)
```

2. Set environment variable:
```bash
export OPENROUTER_API_KEY="your_key"
```

3. Run classification as normal

### Combining Results from Multiple Searches

```python
import json
from glob import glob

# Load all classified papers
all_papers = []
for file in glob("*_papers.json"):
    with open(file) as f:
        papers = json.load(f)
        all_papers.extend(papers)

# Deduplicate by paper ID
unique_papers = {p['paperId']: p for p in all_papers if p.get('paperId')}

print(f"Found {len(unique_papers)} unique papers")

# Filter for relevant papers
relevant = [
    p for p in unique_papers.values()
    if p.get('relevant', False)
]

print(f"Found {len(relevant)} relevant papers")

# Save combined results
with open("combined_relevant.json", 'w') as f:
    json.dump(relevant, f, indent=2)
```

### Custom Filtering

```python
import json

# Load classified papers
with open("classified_papers.json") as f:
    papers = json.load(f)

# Filter for relevant papers from recent years with high citations
high_confidence_relevant = [
    p for p in papers
    if p.get('relevant', False)
    and p.get('year', 0) >= 2020
    and p.get('citationCount', 0) >= 20
]

print(f"Found {len(high_confidence_relevant)} high-confidence relevant papers")

# Sort by citations
high_confidence_relevant.sort(key=lambda p: p.get('citationCount', 0), reverse=True)

# Save
with open("top_papers.json", 'w') as f:
    json.dump(high_confidence_relevant[:10], f, indent=2)
```

### Running Complete Workflow

Use the provided script:
```bash
bash run.sh
```

Or manually:
```bash
python asta.py && python classify_papers.py && python download_papers.py
```

## Project Structure

```
GetRelevantPapers/
├── README.md                    # This file
├── config.yaml                  # Hydra configuration file
├── requirements.txt             # Python dependencies
│
├── MAIN WORKFLOW SCRIPTS (Root Level):
├── asta.py                      # Search papers using ASTA Corpus
├── semantic_scholar.py          # Search papers using Semantic Scholar
├── classify_papers.py           # Classify papers using LLM classifiers
├── download_papers.py           # Download PDFs from arXiv
├── visualize_papers.py          # Visualize classified papers
├── main.py                      # Main analysis script
├── txt_to_markdown.py           # Convert text to markdown
│
├── src/                         # Source code
│   ├── core/                    # Core functionality
│   │   ├── analyzer.py          # Regex-based paper analyzer
│   │   ├── markdown_parser.py   # Markdown parsing & document DB
│   │   └── vector_store.py      # Vector embeddings & search
│   │
│   └── evaluators/              # Analysis evaluators
│       ├── llm_evaluator.py     # OpenRouter LLM evaluator
│       ├── free_llm_evaluator.py # Free LLM evaluator
│       └── bedrock_evaluator.py # AWS Bedrock evaluator
│
├── src/classifiers/             # Paper classifiers
│   ├── base_classifier.py
│   ├── bedrock_classifier.py
│   └── openrouter_classifier.py
│
├── ui/                          # User interfaces
│   ├── minimal_web_ui.py        # Main web UI (port 3444)
│   ├── simple_ui.py             # Terminal UI
│   ├── web_ui.py                # Full web UI
│   ├── results_web_ui.py        # Results viewer
│   └── realtime_analysis_ui.py  # Real-time analysis UI
│
├── scripts/                     # Utility scripts
│   ├── realtime_analysis.py     # Real-time analysis
│   ├── run_free_llm_analysis.py # Free LLM analysis
│   ├── quick_start.sh           # Quick start script
│   └── run.sh                   # Workflow script
│
├── data/                        # Data storage
│   ├── markdown_db/             # Parsed document database
│   ├── vector_store/            # Vector embeddings
│   ├── converted_papers/        # Converted markdown papers
│   └── classified_papers.json   # Classification results
│
├── docs/                        # Documentation
│   ├── QUICKSTART.md
│   ├── PIPELINE_OVERVIEW.md
│   ├── BEDROCK_SETUP.md
│   ├── DEPLOYMENT_SUMMARY.md
│   └── ...
│
├── archive/                     # Archived data
│   ├── old_analysis_results/    # Historical analysis results
│   └── outputs/                 # Old output logs
│
├── infrastructure/              # Deployment
│   ├── Dockerfile
│   └── terraform/               # Terraform configs
│
└── unimportant/                 # Test/deprecated files
    ├── test_papers/
    └── ...
```

## Files

**Main Workflow Scripts (Root Level):**
- `asta.py` - Search papers from ASTA Corpus
- `semantic_scholar.py` - Search papers from Semantic Scholar  
- `classify_papers.py` - Classify papers using LLM
- `download_papers.py` - Download PDFs from arXiv
- `visualize_papers.py` - Visualize classified papers
- `main.py` - Main analysis pipeline
- `txt_to_markdown.py` - Convert text to markdown

**Core Components:**
- `src/core/` - Core analysis functionality
- `src/evaluators/` - LLM-based evaluators
- `src/classifiers/` - Paper classifiers
- `ui/` - User interfaces (web + terminal)
- `scripts/` - Utility scripts

**Data:**
- `data/` - Databases and converted papers
- `archive/` - Historical results
- `docs/` - Documentation

## User Interfaces

This project provides multiple user interfaces for interacting with the paper analysis pipeline:

### Web UI (Recommended)

The main web interface provides a modern, real-time dashboard for paper analysis.

#### Starting the Web UI

```bash
python src/ui/minimal_web_ui.py
```

**Default Port:** 3444  
**Access URL:** http://localhost:3444

#### Features

- **Real-time Analysis Dashboard**
  - Live progress tracking for paper analysis
  - Visual status indicators for each paper
  - Criteria evaluation with tooltips showing evidence
  - Overall statistics and completion metrics

- **Multiple Analysis Methods**
  - **Regex Analysis**: Fast pattern-based evaluation
  - **Bedrock Analysis (Parallel)**: AWS Bedrock with parallel processing - 1 mega-prompt per paper
  - **Bedrock Analysis (Separate)**: AWS Bedrock with separate criterion calls - 5 API calls per paper
  - **LLM Analysis**: OpenRouter-based evaluation (if configured)

- **Paper Management**
  - Browse all papers in the database
  - View paper details and sections
  - Analyze individual papers on demand

- **Results Viewing**
  - Summary statistics (Include/Exclude/Review counts)
  - Detailed criteria breakdown for each paper
  - Score-based recommendations

#### Usage

1. Start the web UI:
```bash
python src/ui/minimal_web_ui.py
```

2. Open http://localhost:3444 in your browser

3. Click "Analyze All" with your preferred method:
   - **Regex**: Fastest, uses pattern matching
   - **Bedrock (Parallel)** 🚀: Most accurate, uses AWS AI with 1 mega-prompt per paper (requires AWS credentials)
   - **Bedrock (Separate)** 🔬: AWS AI with 5 separate criterion calls per paper - more granular progress tracking
   - **LLM**: Alternative AI-based analysis (requires OpenRouter API key)

4. Watch real-time progress as papers are analyzed

5. Review results with detailed criteria evaluation and evidence

#### Bedrock Analysis Methods Comparison

The Web UI offers two Bedrock analysis approaches:

| Feature | Parallel (🚀) | Separate (🔬) |
|---------|--------------|---------------|
| **API Calls** | 1 per paper | 5 per paper |
| **Prompt Type** | Mega-prompt with all criteria | Individual prompts per criterion |
| **Progress Granularity** | Per-paper completion | Per-criterion completion (20%, 40%, 60%, 80%, 100%) |
| **Cost** | Lower (fewer calls) | Higher (5x more calls) |
| **Reasoning** | Cross-criterion coherence | Independent evaluations |
| **Best For** | Production analysis, cost efficiency | Debugging, detailed tracking, comparison |

**When to use Parallel (🚀):**
- Analyzing many papers efficiently
- Lower cost is important
- Want coherent cross-criterion reasoning

**When to use Separate (🔬):**
- Need detailed progress tracking
- Debugging criterion-specific issues
- Comparing evaluation methods
- Want more frequent UI updates

See `ANALYSIS_METHODS_COMPARISON.md` for detailed comparison and `test_separate_criteria.py` to test both methods.

### Terminal UI

A command-line interface with interactive menus for users who prefer terminal-based workflows.

#### Starting the Terminal UI

```bash
python ui/simple_ui.py
```

#### Features

- **Upload/Process Papers**
  - Process single markdown files
  - Batch process directories
  - Convert TXT files to markdown
  - Process sample papers

- **Paper Analysis**
  - Choose between Regex and LLM analysis
  - Analyze all papers or select specific ones
  - View progress and results in terminal

- **Paper Browsing**
  - List all papers in database
  - View detailed paper information
  - Check section contents

- **Results Management**
  - View historical analysis results
  - Browse by date and method
  - See top-ranked papers

- **System Information**
  - Database statistics
  - Vector store status
  - Available analysis methods

#### Usage

```bash
python ui/simple_ui.py
```

Navigate menus using number keys:
- `1` - Upload/Process Papers
- `2` - Analyze Papers (Regex)
- `3` - Analyze Papers (LLM)
- `4` - View Papers
- `5` - View Results
- `6` - Process Sample Papers
- `7` - System Info
- `0` - Exit

### Other UIs

Additional specialized interfaces are available in the `ui/` directory:

- `web_ui.py` - Full-featured web interface with extended functionality
- `results_web_ui.py` - Dedicated results visualization interface
- `realtime_analysis_ui.py` - Streaming analysis with live updates

## Next Steps

After classification:
1. Review classified papers: `python visualize_papers.py classified_papers.json`
2. Filter papers where both models agree for high confidence
3. Download PDFs: `python download_papers.py`
4. Review disagreements manually - they may reveal edge cases
5. Adjust research description in `config.yaml` if needed
6. Use relevant papers as seeds for graph exploration
7. Build reading list from high-confidence relevant papers
