"""
TXT to Markdown Converter

Converts extracted TXT files from PDFs into structured markdown documents
for processing through the analysis pipeline.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import json


class TxtToMarkdownConverter:
    """Convert TXT files to structured markdown"""
    
    def __init__(self):
        # Common section patterns in academic papers
        self.section_patterns = [
            (r'^(?:abstract|summary)\s*$', 'Abstract'),
            (r'^(?:1\.?\s*)?(?:introduction|intro)\s*$', 'Introduction'),
            (r'^(?:2\.?\s*)?(?:related\s*work|background|literature\s*review)\s*$', 'Related Work'),
            (r'^(?:3\.?\s*)?(?:method(?:ology)?|approach|model|framework)\s*$', 'Methodology'),
            (r'^(?:4\.?\s*)?(?:experiment(?:s)?|evaluation|results)\s*$', 'Experiments'),
            (r'^(?:5\.?\s*)?(?:discussion)\s*$', 'Discussion'),
            (r'^(?:6\.?\s*)?(?:conclusion(?:s)?|summary)\s*$', 'Conclusion'),
            (r'^(?:references|bibliography)\s*$', 'References'),
            (r'^(?:appendix|supplementary)\s*$', 'Appendix'),
            # Numbered sections
            (r'^\d+\.?\s+(.+)$', r'\1'),
            # All caps sections
            (r'^([A-Z\s]{3,})\s*$', r'\1'),
        ]
    
    def combine_txt_files(self, paper_dir: Path) -> str:
        """Combine all TXT files from a paper directory"""
        txt_files = sorted(paper_dir.glob("*.txt"))
        
        if not txt_files:
            return ""
        
        combined_text = []
        
        for txt_file in txt_files:
            try:
                content = txt_file.read_text(encoding='utf-8', errors='ignore')
                # Clean up the content
                content = self._clean_text(content)
                combined_text.append(content)
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
                continue
        
        return "\n\n".join(combined_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Leading/trailing spaces
        
        return text
    
    def convert_to_markdown(self, text: str, paper_title: str) -> str:
        """Convert plain text to structured markdown"""
        lines = text.split('\n')
        markdown_lines = [f"# {paper_title}\n"]
        
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if section_content:  # Keep empty lines within sections
                    section_content.append("")
                continue
            
            # Check if this line is a section header
            section_match = self._identify_section(line)
            
            if section_match:
                # Save previous section
                if current_section and section_content:
                    markdown_lines.append(f"\n## {current_section}\n")
                    markdown_lines.extend(section_content)
                    markdown_lines.append("")
                
                current_section = section_match
                section_content = []
            else:
                # Add to current section content
                section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            markdown_lines.append(f"\n## {current_section}\n")
            markdown_lines.extend(section_content)
        
        # If no sections were found, add all content as introduction
        if not any("##" in line for line in markdown_lines):
            markdown_lines.append("\n## Introduction\n")
            markdown_lines.extend(lines[1:])  # Skip title
        
        return '\n'.join(markdown_lines)
    
    def _identify_section(self, line: str) -> Optional[str]:
        """Identify if a line is a section header"""
        line_lower = line.lower().strip()
        original_line = line.strip()
        
        # Skip very long lines (likely not headers)
        if len(line) > 100:
            return None
        
        # Check for markdown headers already present
        if line.startswith('#'):
            return line.lstrip('#').strip()
        
        # Check against patterns
        for pattern, replacement in self.section_patterns:
            match = re.match(pattern, line_lower, re.IGNORECASE)
            if match:
                if isinstance(replacement, str):
                    return replacement
                else:
                    # Use the captured group
                    return match.group(1).title()
        
        # Check if line looks like a header
        if len(line) < 80:
            # All caps sections
            if line.isupper() and len(line) > 3:
                return line.title()
            
            # Numbered sections
            if re.match(r'^\d+\.?\s+[A-Z]', line):
                return re.sub(r'^\d+\.?\s+', '', line).title()
            
            # Short lines that end with no punctuation (likely headers)
            if (len(line) < 50 and 
                not line.endswith('.') and 
                not line.endswith(',') and
                line.count(' ') < 8 and
                any(c.isupper() for c in line)):
                return line.title()
        
        return None
    
    def process_paper_directory(self, paper_dir: Path) -> Optional[str]:
        """Process a single paper directory and return markdown content"""
        if not paper_dir.is_dir():
            return None
        
        # Extract paper title from directory name
        paper_title = paper_dir.name.replace('_', ' ').replace('-', ' ')
        # Remove arXiv ID prefix if present
        paper_title = re.sub(r'^\d{4}\.\d{5}\s*', '', paper_title)
        
        # Combine all TXT files
        combined_text = self.combine_txt_files(paper_dir)
        
        if not combined_text:
            print(f"No text content found in {paper_dir}")
            return None
        
        # Convert to markdown
        markdown_content = self.convert_to_markdown(combined_text, paper_title)
        
        return markdown_content
    
    def process_all_papers(self, sample_papers_dir: Path, output_dir: Path) -> List[Path]:
        """Process all paper directories and save as markdown files"""
        output_dir.mkdir(exist_ok=True)
        
        paper_dirs = [d for d in sample_papers_dir.iterdir() if d.is_dir()]
        converted_files = []
        
        for paper_dir in paper_dirs:
            print(f"Processing: {paper_dir.name}")
            
            markdown_content = self.process_paper_directory(paper_dir)
            
            if markdown_content:
                # Create output filename
                output_filename = f"{paper_dir.name}.md"
                output_path = output_dir / output_filename
                
                # Save markdown file
                output_path.write_text(markdown_content, encoding='utf-8')
                converted_files.append(output_path)
                
                print(f"  ✓ Converted to {output_filename}")
            else:
                print(f"  ✗ Failed to convert {paper_dir.name}")
        
        return converted_files


def main():
    """Convert all sample papers to markdown"""
    converter = TxtToMarkdownConverter()
    
    sample_papers_dir = Path("sample_papers")
    output_dir = Path("converted_papers")
    
    if not sample_papers_dir.exists():
        print(f"Sample papers directory not found: {sample_papers_dir}")
        return
    
    print("Converting TXT files to Markdown...")
    converted_files = converter.process_all_papers(sample_papers_dir, output_dir)
    
    print(f"\n✓ Converted {len(converted_files)} papers to markdown")
    print(f"Output directory: {output_dir}")
    
    # Show first few files
    for i, file_path in enumerate(converted_files[:5]):
        print(f"  {i+1}. {file_path.name}")
    
    if len(converted_files) > 5:
        print(f"  ... and {len(converted_files) - 5} more")


if __name__ == "__main__":
    main()
