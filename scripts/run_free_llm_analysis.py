#!/usr/bin/env python3
"""
Run Free LLM Analysis on All Papers
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unimportant.simple_free_llm import SimpleFreeEvaluator
from src.core.markdown_parser import DocumentDatabase
from src.core.vector_store import SimpleVectorStore

def main():
    """Run free LLM analysis on all papers"""
    print("🆓 Starting FREE LLM Analysis on all papers...")
    
    # Setup
    db_path = Path("data/markdown_db")
    vector_path = Path("data/vector_store")
    
    db = DocumentDatabase(db_path)
    vector_store = SimpleVectorStore(vector_path)
    evaluator = SimpleFreeEvaluator(db, vector_store)
    
    # Get all documents
    docs = db.list_documents()
    print(f"📚 Found {len(docs)} papers to analyze")
    
    # Analyze all papers
    results = []
    for i, doc_info in enumerate(docs):
        print(f"\n📄 Analyzing {i+1}/{len(docs)}: {doc_info['title']}")
        
        result = evaluator.analyze_paper(doc_info['doc_id'])
        if result:
            results.append(result)
            print(f"   ✅ {result.recommendation} (Score: {result.overall_score:.2f})")
        else:
            print(f"   ❌ Failed to analyze")
    
    # Save results
    if results:
        evaluator.save_results(results)
        print(f"\n🎉 Analysis complete! {len(results)} papers analyzed")
        
        # Print summary
        include_count = sum(1 for r in results if r.recommendation == "Include")
        exclude_count = sum(1 for r in results if r.recommendation == "Exclude")
        review_count = sum(1 for r in results if r.recommendation == "Review")
        
        print(f"📊 Summary:")
        print(f"   ✅ Include: {include_count}")
        print(f"   🔍 Review: {review_count}")
        print(f"   ❌ Exclude: {exclude_count}")
        
        # Show top papers
        sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)
        print(f"\n🏆 Top 5 Papers:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"   {i+1}. {result.title} - {result.recommendation} ({result.overall_score:.2f})")
    
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()







