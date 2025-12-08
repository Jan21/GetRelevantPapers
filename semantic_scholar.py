#!/usr/bin/env python3
"""
Minimalist script to search papers on Semantic Scholar using a research description.
"""

import requests
import json
import time
from typing import List, Dict, Optional
import hydra
from omegaconf import DictConfig


def search_papers(
    cfg: DictConfig
) -> List[Dict]:
    """
    Search for papers on Semantic Scholar with pagination support.

    Args:
        cfg: Hydra configuration object

    Returns:
        List of paper dictionaries

    Note:
        - Uses pagination to fetch more than 100 results
        - Adds delay between paginated requests to avoid rate limiting
        - Stops when limit is reached or no more results available
    """
    query = cfg.search.query
    limit = cfg.search.limit
    fields = list(cfg.search.fields)
    url = cfg.search.api_url
    batch_size = cfg.search.batch_size
    delay = cfg.search.delay
    api_key = cfg.search.get('api_key', None)

    all_papers = []
    offset = 0

    # Prepare headers with API key if available
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key
        print(f"Using Semantic Scholar API key from environment variable\n")

    while len(all_papers) < limit:
        # Calculate how many papers to fetch in this batch
        remaining = limit - len(all_papers)
        current_limit = min(batch_size, remaining)

        params = {
            'query': query,
            'limit': current_limit,
            'offset': offset,
            'fields': ','.join(fields)
        }

        try:
            if offset > 0:
                print(f"Fetching papers {offset + 1}-{offset + current_limit}...", end=' ')

            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            papers = data.get('data', [])

            if offset > 0:
                print(f"got {len(papers)} papers")

            if not papers:
                # No more results available
                if offset == 0:
                    print(f"No papers found for query: '{query[:60]}...'")
                else:
                    print(f"No more papers available (total: {len(all_papers)})")
                break

            all_papers.extend(papers)
            offset += len(papers)

            # Check if we've reached the end of available results
            total_available = data.get('total', 0)
            if total_available > 0 and offset >= total_available:
                print(f"Retrieved all available papers (total: {total_available})")
                break

            if len(papers) < current_limit:
                # Got fewer papers than requested, means we're at the end
                print(f"Retrieved all available papers (total: {len(all_papers)})")
                break

            # Add delay between requests to avoid rate limiting
            if len(all_papers) < limit:
                time.sleep(delay)

        except requests.exceptions.RequestException as e:
            print(f"\nError searching papers: {e}")
            break

    if all_papers:
        print(f"Total papers retrieved: {len(all_papers)} for query: '{query[:60]}...'")

    return all_papers


def display_papers(papers: List[Dict], show_abstract: bool = False):
    """
    Display papers in a readable format.

    Args:
        papers: List of paper dictionaries
        show_abstract: Whether to show abstracts (can be very long)
    """
    for i, paper in enumerate(papers, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}] {paper.get('title', 'N/A')}")
        print(f"{'=' * 80}")

        # Year and venue
        year = paper.get('year', 'N/A')
        venue = paper.get('publicationVenue', {})
        venue_name = venue.get('name', 'N/A') if venue else 'N/A'
        print(f"Year: {year} | Venue: {venue_name}")

        # Abstract (optional)
        if show_abstract:
            abstract = paper.get('abstract', 'No abstract available')
            if not abstract:
                print("No abstract available")
                continue
            print(f"\nAbstract:\n{abstract[:500]}...")


def save_papers(papers: List[Dict], filename: str = 'search_results.json'):
    """
    Save papers to a JSON file.

    Args:
        papers: List of paper dictionaries
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(papers, f, indent=2)
    print(f"\nSaved {len(papers)} papers to {filename}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main function for searching papers using Hydra config."""

    # Search papers
    print(f"\nSearching Semantic Scholar for: '{cfg.search.query}'")
    print(f"Limit: {cfg.search.limit}\n")

    papers = search_papers(cfg)

    if not papers:
        print("No papers found.")
        return

    # Display results
    display_papers(papers, show_abstract=cfg.search.show_abstract)

    # Save if requested
    if cfg.search.output:
        save_papers(papers, cfg.search.output)

    print(f"\n{'=' * 80}")
    print(f"Total papers retrieved: {len(papers)}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
