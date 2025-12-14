#!/usr/bin/env python3
"""
Classify papers from a JSON file using AWS Bedrock with Deepseek.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import hydra
from omegaconf import DictConfig

from classifiers import BedrockClassifier


def load_papers(filename: str) -> List[Dict]:
    """
    Load papers from JSON file.

    Args:
        filename: Path to JSON file

    Returns:
        List of paper dictionaries
    """
    with open(filename, 'r') as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers from {filename}")
    return papers


def classify_papers(
    papers: List[Dict],
    cfg: DictConfig
) -> List[Dict]:
    """
    Classify papers using AWS Bedrock with Deepseek.

    Args:
        papers: List of paper dictionaries
        cfg: Hydra configuration object

    Returns:
        List of papers with classification labels added
    """
    # Initialize Bedrock classifier
    bedrock_classifier = BedrockClassifier(cfg)

    classified_papers = []

    print(f"\n{'='*80}")
    print(f"CLASSIFYING {len(papers)} PAPERS WITH: AWS Bedrock (Deepseek)")
    print(f"{'='*80}\n")

    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Unknown')[:60]
        print(f"[{i}/{len(papers)}] {title}...")

        # Create a copy of the paper to add classification labels
        classified_paper = paper.copy()

        try:
            # Classify with Bedrock
            print(f"  Bedrock: ", end='')
            relevant = bedrock_classifier.classify_paper(paper)
            classified_paper['relevant'] = relevant
            print(f"{'✓ YES' if relevant else '✗ NO'}")

        except Exception as e:
            print(f"  ERROR: {e}")
            classified_paper['relevant'] = False
            classified_paper['error'] = str(e)

        classified_papers.append(classified_paper)
        print()

    # Calculate statistics
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total papers: {len(papers)}")

    relevant_count = sum(1 for p in classified_papers if p.get('relevant', False))
    print(f"Relevant: {relevant_count} ({100 * relevant_count / len(papers):.1f}%)")

    print(f"{'='*80}\n")

    return classified_papers


def save_classified_papers(
    papers: List[Dict],
    output_file: str = "classified_papers.json"
):
    """
    Save classified papers with labels to a JSON file.

    Args:
        papers: List of papers with classification labels
        output_file: Output filename
    """
    with open(output_file, 'w') as f:
        json.dump(papers, f, indent=2)
    print(f"Saved {len(papers)} classified papers to {output_file}")


def display_relevant_papers(papers: List[Dict], cfg: DictConfig, limit: int = 10):
    """
    Display summary of relevant papers.

    Args:
        papers: List of paper dictionaries with classification labels
        cfg: Hydra configuration object
        limit: Maximum number to display
    """
    relevant_papers = [p for p in papers if p.get('relevant', False)]

    print(f"\n{'='*80}")
    print(f"RELEVANT PAPERS (showing {min(limit, len(relevant_papers))} of {len(relevant_papers)})")
    print(f"{'='*80}\n")

    for i, paper in enumerate(relevant_papers[:limit], 1):
        title = paper.get('title', 'N/A')
        year = paper.get('year', 'N/A')
        citations = paper.get('citationCount', 0)
        print(f"[{i}] {title}")
        print(f"    Year: {year} | Citations: {citations}\n")

    if len(relevant_papers) > limit:
        print(f"... and {len(relevant_papers) - limit} more papers")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main function using Hydra config."""

    # Load papers from the search output file
    papers = load_papers(cfg.search.output)

    if not papers:
        print("No papers to classify.")
        return

    # Classify papers
    classified_papers = classify_papers(papers, cfg)

    # Save results to a single file with all labels
    output_file = f"{cfg.classification.output_prefix}_papers.json"
    save_classified_papers(classified_papers, output_file)

    # Display relevant papers
    if cfg.classification.display_results and classified_papers:
        display_relevant_papers(classified_papers, cfg, limit=cfg.classification.display_limit)


if __name__ == '__main__':
    main()
