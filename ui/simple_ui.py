#!/usr/bin/env python3
"""
Simple Terminal UI for Markdown Paper Analysis Pipeline

A command-line interface with menus for uploading papers, running analysis,
and viewing results.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from src.core.markdown_parser import MarkdownParser, DocumentDatabase
from src.core.vector_store import SimpleVectorStore, CriteriaAnalyzer
from src.core.analyzer import PaperAnalyzer
from txt_to_markdown import TxtToMarkdownConverter

# Try to import LLM evaluator
try:
    from llm_evaluator import LLMPaperEvaluator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class SimpleUI:
    """Simple terminal-based UI for paper analysis"""
    
    def __init__(self):
        self.setup_components()
        self.txt_converter = TxtToMarkdownConverter()
    
    def setup_components(self):
        """Initialize analysis components"""
        print("🔧 Setting up analysis components...")
        
        self.db_path = Path("data/markdown_db")
        self.vector_path = Path("data/vector_store")
        
        self.parser = MarkdownParser()
        self.db = DocumentDatabase(self.db_path)
        self.vector_store = SimpleVectorStore(self.vector_path)
        self.criteria_analyzer = CriteriaAnalyzer(self.vector_store)
        
        # Initialize analyzers
        self.regex_analyzer = PaperAnalyzer(self.db, self.vector_store)
        
        if LLM_AVAILABLE:
            self.llm_analyzer = LLMPaperEvaluator(self.db, self.vector_store)
        else:
            self.llm_analyzer = None
        
        print("✓ Components initialized")
    
    def show_main_menu(self):
        """Display main menu"""
        while True:
            self.clear_screen()
            print("=" * 60)
            print("📄 PAPER ANALYSIS PIPELINE")
            print("=" * 60)
            
            stats = self.get_system_stats()
            print(f"📊 System Status:")
            print(f"   Documents: {stats['total_documents']}")
            print(f"   Embeddings: {stats['vector_embeddings']}")
            print(f"   LLM Available: {'Yes' if LLM_AVAILABLE else 'No'}")
            print()
            
            print("🎯 Main Menu:")
            print("1. 📁 Upload/Process Papers")
            print("2. 🔍 Analyze Papers (Regex)")
            if LLM_AVAILABLE:
                print("3. 🧠 Analyze Papers (LLM)")
            print("4. 📋 View Papers")
            print("5. 📊 View Results")
            print("6. 🔄 Process Sample Papers")
            print("7. ⚙️  System Info")
            print("0. 🚪 Exit")
            print()
            
            choice = input("Select option: ").strip()
            
            if choice == "1":
                self.upload_menu()
            elif choice == "2":
                self.analyze_menu("regex")
            elif choice == "3" and LLM_AVAILABLE:
                self.analyze_menu("llm")
            elif choice == "4":
                self.view_papers_menu()
            elif choice == "5":
                self.view_results_menu()
            elif choice == "6":
                self.process_sample_papers()
            elif choice == "7":
                self.system_info_menu()
            elif choice == "0":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Press Enter to continue...")
                input()
    
    def upload_menu(self):
        """Upload/process papers menu"""
        self.clear_screen()
        print("📁 UPLOAD/PROCESS PAPERS")
        print("=" * 40)
        print()
        print("Options:")
        print("1. Process single markdown file")
        print("2. Process directory of markdown files")
        print("3. Convert TXT file to markdown")
        print("4. Convert directory of TXT files")
        print("0. Back to main menu")
        print()
        
        choice = input("Select option: ").strip()
        
        if choice == "1":
            self.process_single_markdown()
        elif choice == "2":
            self.process_markdown_directory()
        elif choice == "3":
            self.convert_single_txt()
        elif choice == "4":
            self.convert_txt_directory()
        elif choice == "0":
            return
        else:
            print("❌ Invalid option")
        
        input("Press Enter to continue...")
    
    def process_single_markdown(self):
        """Process a single markdown file"""
        file_path = input("Enter path to markdown file: ").strip()
        path = Path(file_path)
        
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return
        
        if path.suffix.lower() != '.md':
            print("❌ File must have .md extension")
            return
        
        try:
            print(f"📄 Processing: {path.name}")
            doc = self.parser.parse_file(path)
            doc_id = self.db.store_document(doc)
            
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
                self.vector_store.add_sections(sections_data)
            
            print(f"✅ Successfully processed: {doc.title}")
            print(f"   Document ID: {doc_id}")
            print(f"   Sections: {len(doc.sections)}")
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
    
    def process_markdown_directory(self):
        """Process directory of markdown files"""
        dir_path = input("Enter path to directory: ").strip()
        path = Path(dir_path)
        
        if not path.exists() or not path.is_dir():
            print(f"❌ Directory not found: {dir_path}")
            return
        
        md_files = list(path.glob("*.md"))
        if not md_files:
            print("❌ No markdown files found in directory")
            return
        
        print(f"📁 Found {len(md_files)} markdown files")
        
        processed = 0
        for md_file in md_files:
            try:
                print(f"📄 Processing: {md_file.name}")
                doc = self.parser.parse_file(md_file)
                doc_id = self.db.store_document(doc)
                
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
                    self.vector_store.add_sections(sections_data)
                
                processed += 1
                print(f"   ✅ {doc.title} ({len(doc.sections)} sections)")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n✅ Processed {processed}/{len(md_files)} files")
    
    def convert_single_txt(self):
        """Convert single TXT file to markdown"""
        file_path = input("Enter path to TXT file: ").strip()
        path = Path(file_path)
        
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return
        
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            title = path.stem.replace('_', ' ')
            
            markdown_content = self.txt_converter.convert_to_markdown(content, title)
            
            # Save as markdown
            output_path = path.parent / f"{path.stem}.md"
            output_path.write_text(markdown_content, encoding='utf-8')
            
            print(f"✅ Converted to: {output_path}")
            
            # Ask if user wants to process it
            if input("Process the converted file? (y/n): ").lower() == 'y':
                doc = self.parser.parse_file(output_path)
                doc_id = self.db.store_document(doc)
                print(f"✅ Processed: {doc.title} (ID: {doc_id})")
            
        except Exception as e:
            print(f"❌ Error converting file: {e}")
    
    def convert_txt_directory(self):
        """Convert directory of TXT files"""
        dir_path = input("Enter path to directory: ").strip()
        path = Path(dir_path)
        
        if not path.exists() or not path.is_dir():
            print(f"❌ Directory not found: {dir_path}")
            return
        
        output_dir = path / "converted_markdown"
        output_dir.mkdir(exist_ok=True)
        
        converted_files = self.txt_converter.process_all_papers(path, output_dir)
        
        print(f"✅ Converted {len(converted_files)} files to {output_dir}")
        
        # Ask if user wants to process them
        if input("Process all converted files? (y/n): ").lower() == 'y':
            processed = 0
            for md_file in converted_files:
                try:
                    doc = self.parser.parse_file(md_file)
                    doc_id = self.db.store_document(doc)
                    processed += 1
                except Exception as e:
                    print(f"❌ Error processing {md_file.name}: {e}")
            
            print(f"✅ Processed {processed}/{len(converted_files)} converted files")
    
    def analyze_menu(self, method: str):
        """Analysis menu"""
        self.clear_screen()
        print(f"🔍 ANALYZE PAPERS ({method.upper()})")
        print("=" * 40)
        
        papers = self.db.list_documents()
        if not papers:
            print("❌ No papers in database. Upload some papers first.")
            input("Press Enter to continue...")
            return
        
        print(f"📊 Found {len(papers)} papers in database")
        print()
        print("Options:")
        print("1. Analyze all papers")
        print("2. Analyze specific papers")
        print("0. Back to main menu")
        print()
        
        choice = input("Select option: ").strip()
        
        if choice == "1":
            self.run_analysis(method, None)
        elif choice == "2":
            self.select_and_analyze(method)
        elif choice == "0":
            return
        else:
            print("❌ Invalid option")
        
        input("Press Enter to continue...")
    
    def select_and_analyze(self, method: str):
        """Select specific papers to analyze"""
        papers = self.db.list_documents()
        
        print("\n📋 Available Papers:")
        for i, paper in enumerate(papers, 1):
            print(f"{i:2d}. {paper['title'][:60]}...")
        
        print("\nEnter paper numbers (comma-separated, e.g., 1,3,5):")
        selection = input("Papers to analyze: ").strip()
        
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_papers = [papers[i]['doc_id'] for i in indices if 0 <= i < len(papers)]
            
            if selected_papers:
                self.run_analysis(method, selected_papers)
            else:
                print("❌ No valid papers selected")
        except ValueError:
            print("❌ Invalid input format")
    
    def run_analysis(self, method: str, paper_ids: Optional[List[str]]):
        """Run analysis"""
        print(f"\n🔍 Running {method.upper()} analysis...")
        
        if method == 'llm' and not self.llm_analyzer:
            print("❌ LLM analyzer not available")
            return
        
        analyzer = self.llm_analyzer if method == 'llm' else self.regex_analyzer
        
        try:
            if paper_ids:
                results = []
                for doc_id in paper_ids:
                    print(f"   Analyzing: {doc_id}")
                    result = analyzer.analyze_paper(doc_id)
                    if result:
                        results.append(result)
            else:
                if hasattr(analyzer, 'analyze_all_papers'):
                    results = analyzer.analyze_all_papers()
                else:
                    results = analyzer.analyze_all()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"{method}_analysis_{timestamp}.json")
            analyzer.save_results(results, output_file)
            
            # Show summary
            include_count = sum(1 for r in results if r.recommendation == "Include")
            exclude_count = sum(1 for r in results if r.recommendation == "Exclude")
            review_count = sum(1 for r in results if r.recommendation == "Review")
            
            print(f"\n✅ Analysis Complete!")
            print(f"   Total papers: {len(results)}")
            print(f"   Include: {include_count}")
            print(f"   Exclude: {exclude_count}")
            print(f"   Review: {review_count}")
            print(f"   Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    def view_papers_menu(self):
        """View papers menu"""
        self.clear_screen()
        print("📋 VIEW PAPERS")
        print("=" * 40)
        
        papers = self.db.list_documents()
        if not papers:
            print("❌ No papers in database")
            input("Press Enter to continue...")
            return
        
        print(f"📊 Total papers: {len(papers)}")
        print()
        
        for i, paper in enumerate(papers, 1):
            print(f"{i:2d}. {paper['title']}")
            print(f"     Sections: {paper['section_count']} | ID: {paper['doc_id'][:8]}...")
            print()
        
        # Ask if user wants to view a specific paper
        try:
            choice = input("Enter paper number to view details (or Enter to go back): ").strip()
            if choice:
                idx = int(choice) - 1
                if 0 <= idx < len(papers):
                    self.view_paper_details(papers[idx]['doc_id'])
        except ValueError:
            pass
        
        input("Press Enter to continue...")
    
    def view_paper_details(self, doc_id: str):
        """View detailed paper information"""
        doc_meta = self.db.get_document(doc_id)
        sections = self.db.get_document_sections(doc_id)
        
        if not doc_meta or not sections:
            print("❌ Paper not found")
            return
        
        self.clear_screen()
        print(f"📄 PAPER DETAILS")
        print("=" * 60)
        print(f"Title: {doc_meta['title']}")
        print(f"File: {doc_meta['file_path']}")
        print(f"Sections: {len(sections)}")
        print(f"Parsed: {doc_meta.get('parsed_at', 'Unknown')}")
        print()
        
        print("📑 Sections:")
        for section_key, content in sections.items():
            print(f"  • {section_key.replace('_', ' ').title()}")
            preview = content[:100].replace('\n', ' ')
            print(f"    {preview}...")
            print()
        
        input("Press Enter to continue...")
    
    def view_results_menu(self):
        """View analysis results"""
        self.clear_screen()
        print("📊 VIEW RESULTS")
        print("=" * 40)
        
        # Find result files
        result_files = list(Path("archive/old_analysis_results").glob("*_analysis_*.json"))
        
        if not result_files:
            print("❌ No analysis results found")
            input("Press Enter to continue...")
            return
        
        # Sort by modification time
        result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        print("📁 Available Results:")
        for i, file in enumerate(result_files[:10], 1):
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"{i:2d}. {file.name} ({mtime.strftime('%Y-%m-%d %H:%M')})")
        
        try:
            choice = input("\nEnter result number to view (or Enter to go back): ").strip()
            if choice:
                idx = int(choice) - 1
                if 0 <= idx < len(result_files):
                    self.show_analysis_results(result_files[idx])
        except ValueError:
            pass
        
        input("Press Enter to continue...")
    
    def show_analysis_results(self, result_file: Path):
        """Show detailed analysis results"""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            summary = data.get('summary', {})
            
            self.clear_screen()
            print(f"📊 ANALYSIS RESULTS: {result_file.name}")
            print("=" * 60)
            
            print(f"📈 Summary:")
            print(f"   Total: {summary.get('total', len(results))}")
            print(f"   Include: {summary.get('include', 0)}")
            print(f"   Exclude: {summary.get('exclude', 0)}")
            print(f"   Review: {summary.get('review', 0)}")
            print(f"   Method: {summary.get('method', 'Unknown')}")
            print()
            
            # Show top results
            print("🏆 Top Results:")
            sorted_results = sorted(results, key=lambda x: x.get('overall_score', 0), reverse=True)
            
            for i, result in enumerate(sorted_results[:10], 1):
                score = result.get('overall_score', 0)
                recommendation = result.get('recommendation', 'Unknown')
                title = result.get('title', 'Unknown')
                
                icon = "✅" if recommendation == "Include" else "❌" if recommendation == "Exclude" else "⚠️"
                print(f"{i:2d}. {icon} {title[:50]}...")
                print(f"     Score: {score:.2f} | {recommendation}")
                print()
            
        except Exception as e:
            print(f"❌ Error reading results: {e}")
    
    def process_sample_papers(self):
        """Process sample papers from sample_papers directory"""
        self.clear_screen()
        print("🔄 PROCESS SAMPLE PAPERS")
        print("=" * 40)
        
        sample_dir = Path("sample_papers")
        if not sample_dir.exists():
            print("❌ Sample papers directory not found")
            input("Press Enter to continue...")
            return
        
        print("This will:")
        print("1. Convert TXT files to markdown")
        print("2. Process through analysis pipeline")
        print("3. Run regex analysis")
        print()
        
        if input("Continue? (y/n): ").lower() != 'y':
            return
        
        try:
            # Convert TXT to markdown
            print("📝 Converting TXT files to markdown...")
            output_dir = Path("data/converted_papers")
            converted_files = self.txt_converter.process_all_papers(sample_dir, output_dir)
            print(f"✅ Converted {len(converted_files)} papers")
            
            # Process through pipeline
            print("\n📄 Processing through pipeline...")
            processed = 0
            all_sections_data = []
            
            for md_file in converted_files:
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
            
            print(f"\n✅ Processed {processed} papers")
            
            # Run analysis
            print("\n🔍 Running regex analysis...")
            results = self.regex_analyzer.analyze_all()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"sample_papers_analysis_{timestamp}.json")
            self.regex_analyzer.save_results(results, output_file)
            
            # Show summary
            include_count = sum(1 for r in results if r.recommendation == "Include")
            exclude_count = sum(1 for r in results if r.recommendation == "Exclude")
            review_count = sum(1 for r in results if r.recommendation == "Review")
            
            print(f"\n📊 Analysis Summary:")
            print(f"   Total papers: {len(results)}")
            print(f"   Include: {include_count}")
            print(f"   Exclude: {exclude_count}")
            print(f"   Review: {review_count}")
            print(f"   Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing sample papers: {e}")
        
        input("\nPress Enter to continue...")
    
    def system_info_menu(self):
        """Show system information"""
        self.clear_screen()
        print("⚙️  SYSTEM INFORMATION")
        print("=" * 40)
        
        stats = self.get_system_stats()
        
        print(f"📊 Database:")
        print(f"   Path: {stats['database_path']}")
        print(f"   Documents: {stats['total_documents']}")
        print()
        
        print(f"🧠 Vector Store:")
        print(f"   Path: {stats['vector_store_path']}")
        print(f"   Embeddings: {stats['vector_embeddings']}")
        print()
        
        print(f"🎯 Analysis:")
        print(f"   Criteria: {stats['criteria_count']}")
        print(f"   LLM Available: {stats['llm_available']}")
        print()
        
        print(f"📁 File Structure:")
        for path in [self.db_path, self.vector_path, Path("data/converted_papers")]:
            if path.exists():
                print(f"   ✅ {path}")
            else:
                print(f"   ❌ {path} (missing)")
        
        input("\nPress Enter to continue...")
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_documents": len(self.db.documents),
            "vector_embeddings": self.vector_store.get_stats()["total_embeddings"],
            "criteria_count": len(self.regex_analyzer.criteria),
            "llm_available": LLM_AVAILABLE,
            "database_path": str(self.db_path),
            "vector_store_path": str(self.vector_path)
        }
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')


def main():
    """Main entry point"""
    try:
        ui = SimpleUI()
        ui.show_main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
