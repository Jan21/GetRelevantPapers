#!/usr/bin/env python3
"""
Real-time LLM Analysis with Live Results Display

Shows paper-by-paper analysis results as they complete.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.markdown_parser import MarkdownParser, DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer
from src.core.analyzer import PaperAnalyzer

# Try to import LLM evaluator
try:
    from src.evaluators.llm_evaluator import LLMPaperEvaluator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class RealTimeAnalyzer:
    """Real-time analysis with live progress display"""
    
    def __init__(self):
        self.setup_components()
        self.results = []
        self.current_paper = None
        self.total_papers = 0
        self.completed_papers = 0
    
    def setup_components(self):
        """Initialize analysis components"""
        print("🔧 Setting up analysis components...")
        
        self.db_path = Path("data/markdown_db")
        self.vector_path = Path("data/vector_store")
        
        self.parser = MarkdownParser()
        self.db = DocumentDatabase(self.db_path)
        self.vector_store = SimpleVectorStore(self.vector_path)
        self.criteria_analyzer = CriteriaAnalyzer(self.vector_store)
        self.regex_analyzer = PaperAnalyzer(self.db, self.vector_store)
        
        if LLM_AVAILABLE:
            self.llm_analyzer = LLMPaperEvaluator(self.db, self.vector_store)
        else:
            self.llm_analyzer = None
        
        print("✓ Components initialized")
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def display_progress(self):
        """Display current progress"""
        self.clear_screen()
        print("🧠 REAL-TIME LLM ANALYSIS")
        print("=" * 60)
        print(f"Progress: {self.completed_papers}/{self.total_papers} papers completed")
        
        if self.current_paper:
            print(f"🔍 Currently analyzing: {self.current_paper}")
        
        print("\n📊 COMPLETED RESULTS:")
        print("-" * 60)
        
        # Sort results by score (highest first)
        sorted_results = sorted(self.results, key=lambda x: x.overall_score, reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            # Color coding
            if result.recommendation == "Include":
                icon = "✅"
                color = "\033[92m"  # Green
            elif result.recommendation == "Exclude":
                icon = "❌"
                color = "\033[91m"  # Red
            else:
                icon = "⚠️ "
                color = "\033[93m"  # Yellow
            
            reset_color = "\033[0m"
            
            print(f"{color}{i:2d}. {icon} {result.title[:50]}...{reset_color}")
            print(f"    Score: {result.overall_score:.2f} | {result.recommendation}")
            
            # Show criteria results
            criteria_line = "    "
            for criterion_name, evaluation in result.evaluations.items():
                if evaluation.answer == "Yes":
                    criteria_line += f"✓{criterion_name} "
                elif evaluation.answer == "No":
                    criteria_line += f"✗{criterion_name} "
                else:
                    criteria_line += f"?{criterion_name} "
            
            print(criteria_line)
            print()
        
        if self.completed_papers < self.total_papers:
            print(f"\n🔄 Analysis in progress... {self.total_papers - self.completed_papers} papers remaining")
        else:
            print(f"\n🎉 Analysis complete! {len(self.results)} papers analyzed")
            
            # Show summary
            include_count = sum(1 for r in self.results if r.recommendation == "Include")
            exclude_count = sum(1 for r in self.results if r.recommendation == "Exclude")
            review_count = sum(1 for r in self.results if r.recommendation == "Review")
            
            print(f"\n📈 FINAL SUMMARY:")
            print(f"   ✅ Include: {include_count}")
            print(f"   ❌ Exclude: {exclude_count}")
            print(f"   ⚠️  Review: {review_count}")
            print(f"   📊 Average Score: {sum(r.overall_score for r in self.results) / len(self.results):.2f}")
    
    def analyze_paper_with_progress(self, doc_id: str, method: str = "llm") -> Optional[object]:
        """Analyze single paper and update progress"""
        doc_meta = self.db.get_document(doc_id)
        if not doc_meta:
            return None
        
        self.current_paper = doc_meta['title']
        self.display_progress()
        
        # Choose analyzer
        if method == "llm" and self.llm_analyzer:
            analyzer = self.llm_analyzer
        else:
            analyzer = self.regex_analyzer
        
        # Run analysis
        result = analyzer.analyze_paper(doc_id)
        
        if result:
            self.results.append(result)
            self.completed_papers += 1
            self.display_progress()
        
        return result
    
    def run_realtime_analysis(self, method: str = "llm", paper_ids: Optional[List[str]] = None):
        """Run real-time analysis with live updates"""
        
        # Get papers to analyze
        if paper_ids:
            papers = [{"doc_id": pid} for pid in paper_ids]
        else:
            papers = self.db.list_documents()
        
        if not papers:
            print("❌ No papers found in database")
            return
        
        self.total_papers = len(papers)
        self.completed_papers = 0
        self.results = []
        
        print(f"🚀 Starting real-time {method.upper()} analysis of {self.total_papers} papers...")
        time.sleep(2)
        
        # Analyze each paper
        for paper in papers:
            try:
                self.analyze_paper_with_progress(paper['doc_id'], method)
                time.sleep(0.5)  # Brief pause between papers
            except Exception as e:
                print(f"❌ Error analyzing {paper.get('doc_id', 'unknown')}: {e}")
                continue
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"realtime_{method}_analysis_{timestamp}.json")
        
        if method == "llm" and self.llm_analyzer:
            self.llm_analyzer.save_results(self.results, output_file)
        else:
            self.regex_analyzer.save_results(self.results, output_file)
        
        print(f"\n💾 Results saved to: {output_file}")
        print("\n🎯 Analysis complete! Press Enter to exit...")
        input()
    
    def process_and_analyze_sample_papers(self):
        """Process sample papers and run real-time analysis"""
        print("🔄 PROCESSING SAMPLE PAPERS")
        print("=" * 40)
        
        # First, process the converted papers
        converted_dir = Path("data/converted_papers")
        if not converted_dir.exists():
            print("❌ Converted papers directory not found")
            return
        
        print("📄 Processing converted papers into database...")
        
        md_files = list(converted_dir.glob("*.md"))
        processed = 0
        all_sections_data = []
        
        for md_file in md_files:
            try:
                doc = self.parser.parse_file(md_file)
                doc_id = self.db.store_document(doc)
                
                # Prepare for vector store
                for section_key, content in doc.sections.items():
                    all_sections_data.append({
                        'doc_id': doc_id,
                        'document_title': doc.title,
                        'section_key': section_key,
                        'section_heading': section_key.replace('_', ' ').title(),
                        'content': content
                    })
                
                processed += 1
                print(f"   ✅ {doc.title}")
                
            except Exception as e:
                print(f"   ❌ Error processing {md_file.name}: {e}")
        
        # Add to vector store
        if all_sections_data:
            self.vector_store.add_sections(all_sections_data)
        
        print(f"\n✅ Processed {processed} papers into database")
        print(f"📊 Total sections added to vector store: {len(all_sections_data)}")
        
        # Now run real-time analysis
        print(f"\n🧠 Starting real-time LLM analysis...")
        time.sleep(2)
        
        self.run_realtime_analysis("llm")


def main():
    """Main entry point"""
    analyzer = RealTimeAnalyzer()
    
    print("🎯 REAL-TIME PAPER ANALYSIS")
    print("=" * 40)
    print()
    print("Options:")
    print("1. Process sample papers and analyze with LLM")
    print("2. Analyze existing papers with LLM")
    print("3. Analyze existing papers with Regex")
    print("0. Exit")
    print()
    
    choice = input("Select option: ").strip()
    
    if choice == "1":
        analyzer.process_and_analyze_sample_papers()
    elif choice == "2":
        if not LLM_AVAILABLE:
            print("❌ LLM analyzer not available")
        else:
            analyzer.run_realtime_analysis("llm")
    elif choice == "3":
        analyzer.run_realtime_analysis("regex")
    elif choice == "0":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid option")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
