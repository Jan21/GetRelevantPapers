#!/usr/bin/env python3
"""
Classify papers from a JSON file using dual classifiers (VLLM + OpenRouter).
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import hydra
from omegaconf import DictConfig

from classifiers import VLLMClassifier, OpenRouterClassifier


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
    Classify papers using configured classifiers (VLLM, OpenRouter, or both).

    Args:
        papers: List of paper dictionaries
        cfg: Hydra configuration object

    Returns:
        List of papers with classification labels added
    """
    # Check which classifiers are enabled
    use_vllm = cfg.classification.get('use_vllm', True)
    use_openrouter = cfg.classification.get('use_openrouter', True)

    if not use_vllm and not use_openrouter:
        raise ValueError("At least one classifier must be enabled. Set use_vllm or use_openrouter to true in config.")

    # Initialize enabled classifiers
    vllm_classifier = VLLMClassifier(cfg) if use_vllm else None
    openrouter_classifier = OpenRouterClassifier(cfg) if use_openrouter else None

    classified_papers = []

    # Build classifier list string for display
    classifier_names = []
    if use_vllm:
        classifier_names.append("VLLM")
    if use_openrouter:
        classifier_names.append("OpenRouter")
    classifier_str = " + ".join(classifier_names)

    print(f"\n{'='*80}")
    print(f"CLASSIFYING {len(papers)} PAPERS WITH: {classifier_str}")
    print(f"{'='*80}\n")

    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Unknown')[:60]
        print(f"[{i}/{len(papers)}] {title}...")

        # Create a copy of the paper to add classification labels
        classified_paper = paper.copy()

        try:
            # Classify with VLLM if enabled
            if use_vllm:
                print(f"  VLLM: ", end='')
                vllm_relevant = vllm_classifier.classify_paper(paper)
                classified_paper['vllm_relevant'] = vllm_relevant
                print(f"{'✓ YES' if vllm_relevant else '✗ NO'}")

            # Classify with OpenRouter if enabled
            if use_openrouter:
                print(f"  OpenRouter: ", end='')
                openrouter_relevant = openrouter_classifier.classify_paper(paper)
                classified_paper['openrouter_relevant'] = openrouter_relevant
                print(f"{'✓ YES' if openrouter_relevant else '✗ NO'}")

            # Add agreement flag only if both classifiers are used
            if use_vllm and use_openrouter:
                classified_paper['models_agree'] = (vllm_relevant == openrouter_relevant)
                if classified_paper['models_agree']:
                    print(f"  Agreement: ✓")
                else:
                    print(f"  Agreement: ✗ DISAGREE")

        except Exception as e:
            print(f"  ERROR: {e}")
            if use_vllm:
                classified_paper['vllm_relevant'] = False
            if use_openrouter:
                classified_paper['openrouter_relevant'] = False
            if use_vllm and use_openrouter:
                classified_paper['models_agree'] = True
            classified_paper['error'] = str(e)

        classified_papers.append(classified_paper)
        print()

    # Calculate statistics
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total papers: {len(papers)}")

    if use_vllm:
        vllm_relevant_count = sum(1 for p in classified_papers if p.get('vllm_relevant', False))
        print(f"VLLM relevant: {vllm_relevant_count} ({100 * vllm_relevant_count / len(papers):.1f}%)")

    if use_openrouter:
        openrouter_relevant_count = sum(1 for p in classified_papers if p.get('openrouter_relevant', False))
        print(f"OpenRouter relevant: {openrouter_relevant_count} ({100 * openrouter_relevant_count / len(papers):.1f}%)")

    if use_vllm and use_openrouter:
        both_relevant = sum(1 for p in classified_papers if p.get('vllm_relevant', False) and p.get('openrouter_relevant', False))
        agreement_count = sum(1 for p in classified_papers if p.get('models_agree', False))
        print(f"Both relevant: {both_relevant} ({100 * both_relevant / len(papers):.1f}%)")
        print(f"Models agree: {agreement_count} ({100 * agreement_count / len(papers):.1f}%)")

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
    Display summary of relevant papers based on enabled classifiers.

    Args:
        papers: List of paper dictionaries with classification labels
        cfg: Hydra configuration object
        limit: Maximum number to display
    """
    use_vllm = cfg.classification.get('use_vllm', True)
    use_openrouter = cfg.classification.get('use_openrouter', True)

    # Determine which papers are relevant based on enabled classifiers
    if use_vllm and use_openrouter:
        # Both classifiers enabled: show papers where both agree
        relevant_papers = [p for p in papers if p.get('vllm_relevant', False) and p.get('openrouter_relevant', False)]
        header = "PAPERS RELEVANT BY BOTH MODELS"
    elif use_vllm:
        # Only VLLM enabled
        relevant_papers = [p for p in papers if p.get('vllm_relevant', False)]
        header = "PAPERS RELEVANT BY VLLM"
    else:
        # Only OpenRouter enabled
        relevant_papers = [p for p in papers if p.get('openrouter_relevant', False)]
        header = "PAPERS RELEVANT BY OPENROUTER"

    print(f"\n{'='*80}")
    print(f"{header} (showing {min(limit, len(relevant_papers))} of {len(relevant_papers)})")
    print(f"{'='*80}\n")

    for i, paper in enumerate(relevant_papers[:limit], 1):
        title = paper.get('title', 'N/A')
        year = paper.get('year', 'N/A')
        citations = paper.get('citationCount', 0)
        print(f"[{i}] {title}")
        print(f"    Year: {year} | Citations: {citations}\n")

    if len(relevant_papers) > limit:
        print(f"... and {len(relevant_papers) - limit} more papers")

    # Show disagreements only if both classifiers are enabled
    if use_vllm and use_openrouter:
        disagreements = [p for p in papers if not p.get('models_agree', False)]
        if disagreements:
            print(f"\n{'='*80}")
            print(f"DISAGREEMENTS: {len(disagreements)} papers")
            print(f"{'='*80}\n")
            for i, paper in enumerate(disagreements[:5], 1):
                title = paper.get('title', 'N/A')
                vllm = '✓' if paper.get('vllm_relevant', False) else '✗'
                openrouter = '✓' if paper.get('openrouter_relevant', False) else '✗'
                print(f"[{i}] {title[:70]}")
                print(f"    VLLM: {vllm} | OpenRouter: {openrouter}\n")
            if len(disagreements) > 5:
                print(f"... and {len(disagreements) - 5} more disagreements")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main function using Hydra config."""

    # Load papers from the search output file
    papers = load_papers(cfg.search.output)

    if not papers:
        print("No papers to classify.")
        return

    # Classify papers with both models
    classified_papers = classify_papers(papers, cfg)

    # Save results to a single file with all labels
    output_file = f"{cfg.classification.output_prefix}_papers.json"
    save_classified_papers(classified_papers, output_file)

    # Display relevant papers
    if cfg.classification.display_results and classified_papers:
        display_relevant_papers(classified_papers, cfg, limit=cfg.classification.display_limit)


if __name__ == '__main__':
    main()
