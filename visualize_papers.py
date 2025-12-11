#!/usr/bin/env python3
"""
Minimalistic script to visualize classified papers as a table.
"""

import json
import sys
from typing import List, Dict


def load_papers(filename: str) -> List[Dict]:
    """Load papers from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def visualize_table(papers: List[Dict]):
    """Display papers as a simple table, grouped by classification."""
    if not papers:
        print("No papers to display.")
        return

    # Check if papers have classification labels
    has_labels = any('relevant' in p for p in papers)

    if has_labels:
        # Group papers by classification
        relevant = [p for p in papers if p.get('relevant', False)]
        not_relevant = [p for p in papers if not p.get('relevant', False)]

        # Reorder: relevant first, then not relevant
        ordered_papers = relevant + not_relevant

        # Print header with classification column
        print(f"\n{'='*120}")
        print(f"{'#':<4} {'Year':<6} {'Cites':<6} {'Rel':<5} {'Title':<95}")
        print(f"{'='*120}")

        # Print section headers and rows
        current_section = None
        for i, paper in enumerate(ordered_papers, 1):
            # Determine which section we're in
            if paper in relevant and current_section != 'relevant':
                print(f"\n{'--- RELEVANT ---':^120}\n")
                current_section = 'relevant'
            elif paper in not_relevant and current_section != 'not_relevant':
                print(f"\n{'--- NOT RELEVANT ---':^120}\n")
                current_section = 'not_relevant'

            title = paper.get('title', 'N/A')[:92] + '...' if len(paper.get('title', '')) > 95 else paper.get('title', 'N/A')
            year = str(paper.get('year', 'N/A'))
            citations = str(paper.get('citationCount', 0))
            rel = '✓' if paper.get('relevant', False) else '✗'

            print(f"{i:<4} {year:<6} {citations:<6} {rel:<5} {title:<95}")

        print(f"{'='*120}")

        # Print statistics
        relevant_count = sum(1 for p in papers if p.get('relevant', False))

        print(f"\nTotal papers: {len(papers)}")
        print(f"Relevant: {relevant_count} ({100 * relevant_count / len(papers):.1f}%)")
        print(f"Not relevant: {len(not_relevant)} ({100 * len(not_relevant) / len(papers):.1f}%)\n")

    else:
        # Print header without classification columns
        print(f"\n{'='*120}")
        print(f"{'#':<4} {'Year':<6} {'Cites':<6} {'Title':<100}")
        print(f"{'='*120}")

        # Print rows
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'N/A')[:97] + '...' if len(paper.get('title', '')) > 100 else paper.get('title', 'N/A')
            year = str(paper.get('year', 'N/A'))
            citations = str(paper.get('citationCount', 0))

            print(f"{i:<4} {year:<6} {citations:<6} {title:<100}")

        print(f"{'='*120}")
        print(f"\nTotal papers: {len(papers)}\n")


def main():
    """Main function."""
    filename = sys.argv[1] if len(sys.argv) > 1 else 'classified_papers.json'

    try:
        papers = load_papers(filename)
        visualize_table(papers)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{filename}'.")
        sys.exit(1)


if __name__ == '__main__':
    main()
