#!/usr/bin/env python3
"""
Main script to test the markdown paper analysis system

Usage:
    python main.py --input-dir /path/to/markdown/papers
    python main.py --single-file /path/to/paper.md
    python main.py --analyze-existing  # Analyze papers already in DB
"""

import argparse
from pathlib import Path
import sys

from markdown_parser import MarkdownParser, DocumentDatabase
from vector_store import SimpleVectorStore, CriteriaAnalyzer
from analyzer import PaperAnalyzer

# Try to import LLM evaluator
try:
    from llm_evaluator import LLMPaperEvaluator
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"LLM evaluator not available: {e}")
    LLM_AVAILABLE = False


def setup_system(use_llm=False):
    """Initialize all components"""
    print("Setting up system...")
    
    # Create directories
    db_path = Path("markdown_db")
    vector_path = Path("vector_store")
    
    # Initialize components
    parser = MarkdownParser()
    db = DocumentDatabase(db_path)
    vector_store = SimpleVectorStore(vector_path)
    
    # Choose analyzer type
    if use_llm and LLM_AVAILABLE:
        analyzer = LLMPaperEvaluator(db, vector_store)
        print(f"✓ Using LLM evaluator with model: {analyzer.openrouter.model}")
    else:
        analyzer = PaperAnalyzer(db, vector_store)
        print("✓ Using regex-based analyzer")
    
    print(f"✓ Database: {db_path}")
    print(f"✓ Vector store: {vector_path}")
    print(f"✓ Loaded {len(db.documents)} documents")
    print(f"✓ Vector store stats: {vector_store.get_stats()}")
    
    return parser, db, vector_store, analyzer


def process_markdown_files(parser, db, vector_store, input_path):
    """Process markdown files and add to database and vector store"""
    
    if input_path.is_file():
        # Single file
        print(f"Processing single file: {input_path}")
        doc = parser.parse_file(input_path)
        doc_id = db.store_document(doc)
        
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
            vector_store.add_sections(sections_data)
        
        print(f"✓ Processed: {doc.title}")
        return [doc_id]
    
    elif input_path.is_dir():
        # Directory of files
        print(f"Processing directory: {input_path}")
        docs = parser.parse_directory(input_path)
        
        doc_ids = []
        all_sections_data = []
        
        for doc in docs:
            doc_id = db.store_document(doc)
            doc_ids.append(doc_id)
            
            # Prepare sections for vector store
            for section_key, content in doc.sections.items():
                all_sections_data.append({
                    'doc_id': doc_id,
                    'document_title': doc.title,
                    'section_key': section_key,
                    'section_heading': section_key.replace('_', ' ').title(),
                    'content': content
                })
        
        # Add all sections to vector store
        if all_sections_data:
            vector_store.add_sections(all_sections_data)
        
        print(f"✓ Processed {len(docs)} documents")
        return doc_ids
    
    else:
        print(f"Error: {input_path} not found")
        return []


def analyze_papers(analyzer, doc_ids=None, use_llm=False):
    """Analyze papers against criteria"""
    method = "LLM" if use_llm else "regex"
    print(f"\nAnalyzing papers against criteria using {method}...")
    
    if doc_ids:
        # Analyze specific papers
        results = []
        for doc_id in doc_ids:
            result = analyzer.analyze_paper(doc_id)
            if result:
                results.append(result)
    else:
        # Analyze all papers in database
        if hasattr(analyzer, 'analyze_all_papers'):
            results = analyzer.analyze_all_papers()
        else:
            results = analyzer.analyze_all()
    
    return results


def print_results(results):
    """Print analysis results"""
    if not results:
        print("No results to display")
        return
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS RESULTS ({len(results)} papers)")
    print(f"{'='*60}")
    
    # Summary
    include_count = sum(1 for r in results if r.recommendation == "Include")
    exclude_count = sum(1 for r in results if r.recommendation == "Exclude")
    review_count = sum(1 for r in results if r.recommendation == "Review")
    
    print(f"\nSUMMARY:")
    print(f"  Include: {include_count}")
    print(f"  Exclude: {exclude_count}")
    print(f"  Review:  {review_count}")
    print(f"  Average Score: {sum(r.overall_score for r in results) / len(results):.2f}")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    print(f"{'-'*60}")
    
    # Sort by score (highest first)
    sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   Score: {result.overall_score:.2f} | Recommendation: {result.recommendation}")
        
        # Show criterion results
        for criterion_name, evaluation in result.evaluations.items():
            status = "✓" if evaluation.answer == "Yes" else "✗" if evaluation.answer == "No" else "?"
            print(f"   {status} {criterion_name}: {evaluation.answer} ({evaluation.confidence:.2f})")
        
        print(f"   Evidence: {result.evaluations['pytorch'].evidence[:100]}...")


def analyze_heading_patterns(vector_store):
    """Analyze common heading patterns"""
    print("\nAnalyzing heading patterns...")
    
    patterns = vector_store.analyze_heading_patterns()
    
    print(f"\nMOST COMMON HEADINGS:")
    print(f"{'-'*40}")
    
    for heading, count in list(patterns.items())[:15]:
        print(f"  {count:2d}x  {heading}")


def compare_methods(db, vector_store, doc_ids=None):
    """Compare LLM vs regex analysis methods"""
    if not LLM_AVAILABLE:
        print("LLM evaluator not available for comparison")
        return
    
    print("\nComparing LLM vs Regex analysis methods...")
    
    # Initialize both analyzers
    regex_analyzer = PaperAnalyzer(db, vector_store)
    llm_analyzer = LLMPaperEvaluator(db, vector_store)
    
    # Get documents to analyze
    if doc_ids:
        documents = [{"doc_id": doc_id} for doc_id in doc_ids]
    else:
        documents = db.list_documents()[:3]  # Limit to 3 for comparison
    
    print(f"Comparing analysis on {len(documents)} papers...")
    
    for doc_info in documents:
        doc_id = doc_info['doc_id']
        comparison = llm_analyzer.compare_with_regex(doc_id)
        
        if comparison:
            print(f"\n📄 {comparison['paper_title']}")
            print(f"   LLM Score: {comparison['llm_score']:.2f} | Regex Score: {comparison['regex_score']:.2f}")
            print(f"   LLM: {comparison['llm_recommendation']} | Regex: {comparison['regex_recommendation']}")
            
            # Show criteria differences
            for criterion, comp in comparison['criteria_comparison'].items():
                if not comp['agreement']:
                    print(f"   ❗ {criterion}: LLM={comp['llm_answer']} vs Regex={comp['regex_answer']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze markdown papers against criteria")
    parser.add_argument("--input-dir", type=Path, help="Directory containing markdown files")
    parser.add_argument("--single-file", type=Path, help="Single markdown file to process")
    parser.add_argument("--analyze-existing", action="store_true", help="Analyze papers already in database")
    parser.add_argument("--output", type=Path, default="analysis_results.json", help="Output file for results")
    parser.add_argument("--show-headings", action="store_true", help="Show heading pattern analysis")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-based analysis instead of regex")
    parser.add_argument("--compare-methods", action="store_true", help="Compare LLM vs regex analysis methods")
    parser.add_argument("--openrouter-key", type=str, help="OpenRouter API key (optional for free models)")
    
    args = parser.parse_args()
    
    # Setup system
    md_parser, db, vector_store, analyzer = setup_system(use_llm=args.use_llm)
    
    doc_ids = None
    
    # Process input files if provided
    if args.input_dir:
        doc_ids = process_markdown_files(md_parser, db, vector_store, args.input_dir)
    elif args.single_file:
        doc_ids = process_markdown_files(md_parser, db, vector_store, args.single_file)
    elif not args.analyze_existing:
        print("Error: Must specify --input-dir, --single-file, or --analyze-existing")
        return 1
    
    # Show heading patterns if requested
    if args.show_headings:
        analyze_heading_patterns(vector_store)
    
    # Analyze papers
    results = analyze_papers(analyzer, doc_ids, use_llm=args.use_llm)
    
    # Print results
    print_results(results)
    
    # Save results
    if results:
        analyzer.save_results(results, args.output)
        print(f"\n✓ Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
