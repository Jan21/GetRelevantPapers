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
    has_labels = any('vllm_relevant' in p or 'openrouter_relevant' in p for p in papers)

    if has_labels:
        # Group papers by classification
        both_yes = [p for p in papers if p.get('vllm_relevant', False) and p.get('openrouter_relevant', False)]
        both_no = [p for p in papers if not p.get('vllm_relevant', False) and not p.get('openrouter_relevant', False)]
        disagreements = [p for p in papers if not p.get('models_agree', False)]

        # Reorder: both_yes, disagreements, both_no
        ordered_papers = both_yes + disagreements + both_no

        # Print header with classification columns
        print(f"\n{'='*140}")
        print(f"{'#':<4} {'Year':<6} {'Cites':<6} {'VLLM':<6} {'OR':<6} {'Agree':<6} {'Title':<100}")
        print(f"{'='*140}")

        # Print section headers and rows
        current_section = None
        for i, paper in enumerate(ordered_papers, 1):
            # Determine which section we're in
            if paper in both_yes and current_section != 'both_yes':
                print(f"\n{'--- BOTH MODELS: YES ---':^140}\n")
                current_section = 'both_yes'
            elif paper in disagreements and current_section != 'disagreements':
                print(f"\n{'--- MODELS DISAGREE ---':^140}\n")
                current_section = 'disagreements'
            elif paper in both_no and current_section != 'both_no':
                print(f"\n{'--- BOTH MODELS: NO ---':^140}\n")
                current_section = 'both_no'

            title = paper.get('title', 'N/A')[:97] + '...' if len(paper.get('title', '')) > 100 else paper.get('title', 'N/A')
            year = str(paper.get('year', 'N/A'))
            citations = str(paper.get('citationCount', 0))
            vllm = '✓' if paper.get('vllm_relevant', False) else '✗'
            openrouter = '✓' if paper.get('openrouter_relevant', False) else '✗'
            agree = '✓' if paper.get('models_agree', False) else '✗'

            print(f"{i:<4} {year:<6} {citations:<6} {vllm:<6} {openrouter:<6} {agree:<6} {title:<100}")

        print(f"{'='*140}")

        # Print statistics
        vllm_count = sum(1 for p in papers if p.get('vllm_relevant', False))
        or_count = sum(1 for p in papers if p.get('openrouter_relevant', False))
        agree_count = sum(1 for p in papers if p.get('models_agree', False))

        print(f"\nTotal papers: {len(papers)}")
        print(f"Both YES: {len(both_yes)} ({100 * len(both_yes) / len(papers):.1f}%)")
        print(f"Disagreements: {len(disagreements)} ({100 * len(disagreements) / len(papers):.1f}%)")
        print(f"Both NO: {len(both_no)} ({100 * len(both_no) / len(papers):.1f}%)")
        print(f"VLLM relevant: {vllm_count} ({100 * vllm_count / len(papers):.1f}%)")
        print(f"OpenRouter relevant: {or_count} ({100 * or_count / len(papers):.1f}%)")
        print(f"Models agree: {agree_count} ({100 * agree_count / len(papers):.1f}%)\n")

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
