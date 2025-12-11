# Markdown Paper Analysis System

A comprehensive system for parsing markdown documents (research papers) and analyzing them against the criteria from the Deep Researcher project.

## Overview

This system implements:

1. **Markdown Parser**: Extracts sections from markdown documents based on headings
2. **Document Database**: Simple disk-based storage for parsed documents  
3. **Vector Store**: TF-IDF based semantic search for section headings
4. **Criteria Analyzer**: Finds relevant sections for each evaluation criterion
5. **Paper Analyzer**: Evaluates papers against Deep Researcher criteria

## Evaluation Criteria

Based on the Deep Researcher project, papers are evaluated against:

- **PyTorch Framework** (Required): Uses PyTorch for implementation
- **Supervised Learning** (Required): Focuses on supervised learning methods  
- **Small Dataset** (Preferred): Works with ≤100K samples (CIFAR-10, MNIST, etc.)
- **Quick Training** (Preferred): Trainable ≤24 hours on single GPU
- **Public Repository** (Required): Provides public code repository

## Files

- `markdown_parser.py` - Parse markdown documents into structured sections
- `vector_store.py` - Simple TF-IDF vector store for semantic search
- `analyzer.py` - Main paper analysis logic with criteria evaluation
- `main.py` - Command-line interface for the system

## Usage

### Process and Analyze Papers

```bash
# Analyze a directory of markdown papers
python main.py --input-dir /path/to/papers --show-headings

# Analyze a single paper
python main.py --single-file paper.md

# Analyze papers already in database
python main.py --analyze-existing

# Save results to custom file
python main.py --input-dir papers --output my_results.json
```

### Example Output

```
ANALYSIS RESULTS (3 papers)
============================================================

SUMMARY:
  Include: 2
  Exclude: 1  
  Review:  0
  Average Score: 0.65

DETAILED RESULTS:
------------------------------------------------------------

1. Efficient Neural Networks for MNIST Classification
   Score: 0.88 | Recommendation: Include
   ✓ pytorch: Yes (0.90)
   ✓ supervised: Yes (0.90) 
   ✓ small_dataset: Yes (0.90)
   ✓ quick_training: Yes (0.70)
   ✓ has_repo: Yes (0.90)
```

## System Architecture

### 1. Markdown Parser (`MarkdownParser`)

- Parses markdown files using regex patterns for headings
- Extracts sections with heading levels and content
- Handles nested sections and creates clean section keys

### 2. Document Database (`DocumentDatabase`)

- Stores parsed documents on disk as JSON
- Tracks document metadata and change detection via file hashes
- Provides search functionality across sections
- Structure:
  ```
  markdown_db/
    documents.json     # Document metadata
    sections/          # Individual section files  
    raw/              # Raw document content
  ```

### 3. Vector Store (`SimpleVectorStore`)

- Creates TF-IDF embeddings for section headings and content
- Enables semantic search to find relevant sections
- Groups similar sections and analyzes heading patterns
- Stores embeddings and vocabulary on disk

### 4. Criteria Analyzer (`CriteriaAnalyzer`)

- Maps evaluation criteria to relevant keywords
- Uses vector store to find sections most relevant to each criterion
- Provides section content ranked by relevance score

### 5. Paper Analyzer (`PaperAnalyzer`)

- Evaluates papers against all criteria using pattern matching
- Uses positive/negative regex patterns for each criterion
- Calculates confidence scores and overall paper scores
- Makes Include/Exclude/Review recommendations

## Criteria Evaluation Logic

For each criterion, the system:

1. **Finds Relevant Sections**: Uses vector store to identify sections most likely to contain evidence
2. **Pattern Matching**: Applies positive and negative regex patterns to text
3. **Scoring**: Counts matches and calculates confidence scores
4. **Decision**: Determines Yes/No/Unknown based on evidence strength

### Example Patterns

**PyTorch Criterion:**
- Positive: `pytorch`, `torch\.`, `torchvision`  
- Negative: `tensorflow`, `keras`, `jax`

**Small Dataset Criterion:**
- Positive: `cifar`, `mnist`, `60,?000.*samples`
- Negative: `imagenet`, `million.*images`, `large.*scale`

## Output Format

Results are saved as JSON with:

```json
{
  "results": [
    {
      "doc_id": "...",
      "title": "Paper Title",
      "overall_score": 0.88,
      "recommendation": "Include",
      "evaluations": {
        "pytorch": {
          "answer": "Yes",
          "confidence": 0.90,
          "evidence": "Found: pytorch, torch.nn"
        }
      }
    }
  ],
  "summary": {
    "total": 3,
    "include": 2,
    "exclude": 1,
    "review": 0
  }
}
```

## Testing

The system includes sample papers for testing:

- `pytorch_classification.md` - Should be **Include** (meets all criteria)
- `pytorch_mnist.md` - Should be **Include** (efficient PyTorch on small dataset)  
- `tensorflow_large_scale.md` - Should be **Exclude** (wrong framework, large dataset)

Run test:
```bash
python main.py --input-dir sample_papers --show-headings
```

## Key Features

✅ **Section-based Analysis**: Analyzes specific document sections rather than full text  
✅ **Semantic Search**: Finds relevant sections using TF-IDF similarity  
✅ **Pattern Matching**: Uses regex patterns for reliable criterion detection  
✅ **Confidence Scoring**: Provides confidence levels for each evaluation  
✅ **Disk Storage**: Persistent storage for documents and embeddings  
✅ **Heading Analysis**: Identifies common section patterns across papers  
✅ **Extensible**: Easy to add new criteria or modify existing ones

## Integration with Deep Researcher

This system implements the same evaluation criteria as the Deep Researcher project:

- Same 5 criteria (pytorch, supervised, small_dataset, quick_training, has_repo)
- Same scoring and recommendation logic  
- Compatible output format for integration
- Focuses on end-to-end solutions rather than components

The key difference is that this system works with **markdown documents** instead of PDFs, making it suitable for analyzing papers that are already in markdown format or have been converted from PDF to markdown.

## Future Enhancements

- Add support for more sophisticated NLP models (BERT, etc.)
- Implement active learning to improve pattern detection
- Add support for custom criteria definitions
- Create web interface for easier paper analysis
- Add integration with paper databases (arXiv, Semantic Scholar)
