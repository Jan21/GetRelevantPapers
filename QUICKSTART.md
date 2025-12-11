# 🚀 GetRelevantPapers - AWS Bedrock Edition

**Status**: ✅ vLLM removed, AWS Bedrock with Deepseek integrated

## What Changed?

- ❌ **REMOVED**: Local vLLM server requirement
- ❌ **REMOVED**: Dual classifier system (vLLM + OpenRouter)
- ✅ **ADDED**: AWS Bedrock integration with Deepseek
- ✅ **SIMPLIFIED**: Single classifier, cleaner output
- ✅ **IMPROVED**: No local GPU needed, AWS manages infrastructure

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure AWS
```bash
aws configure
# Enter your AWS credentials when prompted
```

### 3. Verify Setup
```bash
./quick_start.sh
```

### 4. Run the Pipeline
```bash
# Search for papers
python asta.py

# Classify with Bedrock
python classify_papers.py

# View results
python visualize_papers.py

# Download PDFs
python download_papers.py
```

## Documentation

- **[BEDROCK_SETUP.md](BEDROCK_SETUP.md)** - Complete AWS Bedrock setup guide
- **[MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)** - Detailed migration notes
- **[README.md](README.md)** - Full project documentation

## Key Files

### Core Scripts
- `asta.py` - Search papers from ASTA Corpus
- `semantic_scholar.py` - Search papers from Semantic Scholar  
- `classify_papers.py` - Classify papers with Bedrock (Deepseek)
- `visualize_papers.py` - Display classification results
- `download_papers.py` - Download PDFs from arXiv

### Configuration
- `config.yaml` - All settings (Bedrock, search, download)
- `requirements.txt` - Python dependencies

### Classifiers
- `classifiers/bedrock_classifier.py` - AWS Bedrock integration
- `classifiers/openrouter_classifier.py` - OpenRouter (backup option)
- `classifiers/base_classifier.py` - Base class

## Configuration (config.yaml)

```yaml
# AWS Bedrock with Deepseek
bedrock:
  region: "us-east-1"
  model_id: "us.amazon.nova-micro-v1:0"  # Update with your Deepseek model ID
  max_tokens: 10
  temperature: 0.1

# Classification settings
classification:
  research_description: "Papers that apply graph neural networks to Boolean satisfiability problems..."
  output_prefix: "classified"
  display_results: true
  display_limit: 10

# Paper search
asta:
  search_queries:
    - "graph neural networks for sat solving"
    - "GNN for Boolean satisfiability"
  limit: 100

# PDF download
download:
  output_dir: "downloaded_papers"
  delay: 1.0
```

## Output Files

- `search_results.json` - Papers from ASTA/Semantic Scholar
- `classified_papers.json` - Classification results with `relevant` field
- `downloaded_papers/` - PDFs from arXiv

## Output Format

```json
{
  "paperId": "abc123",
  "title": "Graph Neural Networks for SAT Solving",
  "abstract": "We present a novel approach...",
  "year": 2023,
  "citationCount": 42,
  "relevant": true
}
```

## Requirements

- Python 3.8+
- AWS account with Bedrock access
- AWS credentials configured
- Internet connection

## Cost

AWS Bedrock typical costs:
- ~$0.003 per 1K input tokens
- ~$0.006 per 1K output tokens
- **1000 papers**: ~$2-5 USD

Much cheaper than OpenAI/OpenRouter for bulk processing.

## Troubleshooting

### AWS Credentials Not Found
```bash
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

### Model Access Denied
1. Go to AWS Console → Bedrock → Model access
2. Request access to Deepseek model
3. Wait for approval (usually instant)

### Invalid Model ID
```bash
# List available models
aws bedrock list-foundation-models --region us-east-1

# Update config.yaml with valid model ID
```

## Support

- See [BEDROCK_SETUP.md](BEDROCK_SETUP.md) for detailed setup
- See [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) for migration details
- Check [README.md](README.md) for full documentation

## License

MIT License - see repository for details

