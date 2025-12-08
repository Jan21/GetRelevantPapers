# Search and Classify Workflow

A workflow for finding and classifying relevant papers using multiple search sources and dual LLM-based classification:
1. **Search**: Find papers using Semantic Scholar or ASTA Corpus
2. **Classify**: Filter papers using dual LLM classifiers (VLLM + OpenRouter)
3. **Download**: Download PDFs from arXiv for relevant papers
4. **Visualize**: View classified papers in a table format

## Prerequisites

1. **Python packages**:
   ```bash
   pip install hydra-core omegaconf requests openai
   ```

2. **Environment variables** (optional but recommended):
   ```bash
   export SEMANTIC_SCHOLAR_API_KEY="your_key"  # For Semantic Scholar (optional)
   export ASTA_API_KEY="your_key"              # For ASTA Corpus
   export OPENROUTER_API_KEY="your_key"        # For OpenRouter classifier
   ```

3. **vLLM server** (optional, for local classification):
   ```bash
   vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
       --gpu-memory-utilization 0.9 \
       --max-model-len 2048 \
       --port 8000
   ```

## Configuration

All settings are configured via `config.yaml` using Hydra. Key sections:

- **ASTA**: Search queries and API settings for ASTA Corpus
- **Search**: Semantic Scholar API settings
- **VLLM**: Local vLLM server configuration
- **OpenRouter**: Cloud-based classifier configuration
- **Classification**: Research description and classifier selection
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

Filter papers using dual LLM-based classification (VLLM and/or OpenRouter):

```bash
python classify_papers.py
```

**Configuration** (in `config.yaml`):
- `classification.research_description`: Detailed criteria for relevance
- `classification.use_vllm`: Enable/disable VLLM classifier (default: true)
- `classification.use_openrouter`: Enable/disable OpenRouter classifier (default: true)
- `classification.output_prefix`: Output file prefix (default: "classified")
- `classification.display_results`: Show results summary (default: true)
- `classification.display_limit`: Number of papers to display (default: 10)

**Output**:
- `classified_papers.json` - All papers with classification labels:
  - `vllm_relevant`: Boolean (if VLLM enabled)
  - `openrouter_relevant`: Boolean (if OpenRouter enabled)
  - `models_agree`: Boolean (if both enabled)

### Step 3: Download PDFs (Optional)

Download PDFs from arXiv for papers classified as relevant:

```bash
python download_papers.py
```

**Configuration** (in `config.yaml`):
- `download.output_dir`: Directory to save PDFs (default: "downloaded_papers")
- `download.delay`: Delay between downloads in seconds (default: 1.0)

**Behavior**:
- Only downloads papers where both classifiers agree (if both enabled)
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
    "vllm_relevant": true,
    "openrouter_relevant": true,
    "models_agree": true,
    ...
  },
  {
    "paperId": "def456",
    "title": "Image Classification with Deep Learning",
    ...
    "vllm_relevant": false,
    "openrouter_relevant": false,
    "models_agree": true,
    ...
  }
]
```

### Terminal Output

```
================================================================================
CLASSIFYING 200 PAPERS WITH: VLLM + OpenRouter
================================================================================

[1/200] Graph Neural Networks for SAT Solving...
  VLLM: ✓ YES
  OpenRouter: ✓ YES
  Agreement: ✓

[2/200] Image Classification with Deep Learning...
  VLLM: ✗ NO
  OpenRouter: ✗ NO
  Agreement: ✓

[3/200] NeuroSAT: Learning a SAT Solver...
  VLLM: ✓ YES
  OpenRouter: ✗ NO
  Agreement: ✗ DISAGREE

...

================================================================================
CLASSIFICATION COMPLETE
================================================================================
Total papers: 200
VLLM relevant: 35 (17.5%)
OpenRouter relevant: 42 (21.0%)
Both relevant: 28 (14.0%)
Models agree: 185 (92.5%)
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
Depends on enabled classifiers:
- **VLLM only** (with GPU): ~1-2 seconds per paper
- **OpenRouter only**: ~2-3 seconds per paper
- **Both enabled**: ~3-5 seconds per paper
- **100 papers**: ~5-8 minutes (both enabled)
- **500 papers**: ~25-40 minutes (both enabled)

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

**Problem**: vLLM connection error
**Solution**:
```bash
# Check if vLLM server is running
curl http://0.0.0.0:8000/v1/models

# Restart server if needed
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --port 8000
```

**Problem**: OpenRouter API key error
**Solution**: Set environment variable:
```bash
export OPENROUTER_API_KEY="your_key"
```

**Problem**: All papers classified as irrelevant
**Solution**:
- Make research description more inclusive in `config.yaml`
- Check if papers have abstracts
- Try broader criteria
- Review a few papers manually to understand why they're being rejected

**Problem**: Too many false positives
**Solution**:
- Make research description more specific
- Add exclusion criteria
- Use both classifiers and require agreement (both must say YES)
- Manually review results

**Problem**: Models disagree frequently
**Solution**:
- This is normal - different models have different interpretations
- Review disagreements manually
- Adjust research description to be more explicit
- Consider using only one classifier if disagreements are problematic

## Advanced Usage

### Using Only One Classifier

To use only VLLM (disable OpenRouter):
```bash
python classify_papers.py classification.use_openrouter=false
```

To use only OpenRouter (disable VLLM):
```bash
python classify_papers.py classification.use_vllm=false
```

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

# Filter for papers relevant by both models
both_relevant = [
    p for p in unique_papers.values()
    if p.get('vllm_relevant', False) and p.get('openrouter_relevant', False)
]

print(f"Found {len(both_relevant)} papers relevant by both models")

# Save combined results
with open("combined_relevant.json", 'w') as f:
    json.dump(both_relevant, f, indent=2)
```

### Custom Filtering

```python
import json

# Load classified papers
with open("classified_papers.json") as f:
    papers = json.load(f)

# Filter for papers where both models agree and are relevant
high_confidence_relevant = [
    p for p in papers
    if p.get('vllm_relevant', False) 
    and p.get('openrouter_relevant', False)
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
├── config.yaml              # Hydra configuration file
├── asta.py                  # Search papers using ASTA Corpus
├── semantic_scholar.py      # Search papers using Semantic Scholar
├── classify_papers.py       # Classify papers using dual LLM classifiers
├── download_papers.py       # Download PDFs from arXiv
├── visualize_papers.py      # Visualize classified papers
├── run.sh                   # Example workflow script
├── classifiers/             # Classifier implementations
│   ├── base_classifier.py   # Base class for classifiers
│   ├── vllm_classifier.py   # VLLM-based classifier
│   └── openrouter_classifier.py  # OpenRouter-based classifier
├── downloaded_papers/        # Directory for downloaded PDFs
├── outputs/                 # Log files from runs
└── README.md                # This file
```

## Files

- `config.yaml` - Main configuration file (Hydra)
- `asta.py` - Search papers from ASTA Corpus
- `semantic_scholar.py` - Search papers from Semantic Scholar
- `classify_papers.py` - Classify papers using VLLM and/or OpenRouter
- `download_papers.py` - Download PDFs from arXiv for relevant papers
- `visualize_papers.py` - Display classified papers in table format
- `run.sh` - Example workflow script
- `classifiers/` - Classifier implementations

## Next Steps

After classification:
1. Review classified papers: `python visualize_papers.py classified_papers.json`
2. Filter papers where both models agree for high confidence
3. Download PDFs: `python download_papers.py`
4. Review disagreements manually - they may reveal edge cases
5. Adjust research description in `config.yaml` if needed
6. Use relevant papers as seeds for graph exploration
7. Build reading list from high-confidence relevant papers
