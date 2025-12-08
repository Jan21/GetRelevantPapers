#!/usr/bin/env python3
"""
Download PDFs from arXiv for papers classified as relevant.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote
import requests
import hydra
from omegaconf import DictConfig


def search_arxiv_by_title(title: str, max_results: int = 5) -> Optional[str]:
    """
    Search arXiv API for a paper by title and return the arXiv ID if found.

    Args:
        title: Paper title to search for
        max_results: Maximum number of results to check

    Returns:
        arXiv ID if found with high confidence, None otherwise
    """
    if not title or len(title.strip()) < 10:
        return None

    try:
        # Clean and encode the title for URL
        search_query = quote(title.strip())
        api_url = f"http://export.arxiv.org/api/query?search_query=ti:{search_query}&max_results={max_results}"

        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        # Parse XML response to find arXiv ID
        content = response.text

        # Extract arXiv IDs from the response
        id_pattern = r'<id>http://arxiv\.org/abs/(\d{4}\.\d{4,5})(v\d+)?</id>'
        title_pattern = r'<title>(.*?)</title>'

        ids = re.findall(id_pattern, content)
        titles = re.findall(title_pattern, content)

        if not ids or not titles:
            return None

        # Skip the first title (it's the feed title)
        titles = titles[1:]

        # Normalize titles for comparison
        search_title_norm = title.strip().lower()

        # Check if any of the returned titles closely match our search
        for arxiv_id_tuple, found_title in zip(ids, titles):
            found_title_norm = found_title.strip().lower()

            # Simple similarity check: if substantial overlap
            if search_title_norm in found_title_norm or found_title_norm in search_title_norm:
                arxiv_id = arxiv_id_tuple[0]  # Get the ID without version
                print(f"    ✓ Found on arXiv: {arxiv_id}")
                return arxiv_id

        return None

    except Exception as e:
        print(f"    ⚠ arXiv search failed: {e}")
        return None


def extract_arxiv_id(url: str) -> Optional[str]:
    """
    Extract arXiv ID from various arXiv URL formats.

    Args:
        url: URL to parse

    Returns:
        arXiv ID if found, None otherwise

    Examples:
        https://arxiv.org/abs/2103.12345 -> 2103.12345
        https://arxiv.org/pdf/2103.12345.pdf -> 2103.12345
        http://arxiv.org/abs/1234.5678v2 -> 1234.5678
    """
    if not url or 'arxiv.org' not in url.lower():
        return None

    # Match various arXiv ID formats
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})(v\d+)?',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})(v\d+)?',
        r'arxiv\.org/abs/([a-z\-]+/\d{7})(v\d+)?',  # Old format
    ]

    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def download_arxiv_pdf(arxiv_id: str, output_dir: Path, paper_title: str = None) -> Optional[str]:
    """
    Download PDF from arXiv.

    Args:
        arxiv_id: arXiv paper ID
        output_dir: Directory to save the PDF
        paper_title: Optional paper title for filename

    Returns:
        Path to downloaded file if successful, None otherwise
    """
    # Construct PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # Create filename
    if paper_title:
        # Sanitize title for filename
        safe_title = re.sub(r'[^\w\s-]', '', paper_title)
        safe_title = re.sub(r'[\s]+', '_', safe_title)
        filename = f"{arxiv_id}_{safe_title[:50]}.pdf"
    else:
        filename = f"{arxiv_id}.pdf"

    output_path = output_dir / filename

    # Skip if already downloaded
    if output_path.exists():
        print(f"  ✓ Already exists: {filename}")
        return str(output_path)

    try:
        print(f"  Downloading from {pdf_url}...")
        response = requests.get(pdf_url, timeout=30, stream=True)
        response.raise_for_status()

        # Save PDF
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  ✓ Saved: {filename}")
        return str(output_path)

    except Exception as e:
        print(f"  ✗ Failed to download: {e}")
        return None


def get_relevant_papers(papers: List[Dict], cfg: DictConfig) -> List[Dict]:
    """
    Filter papers based on classifier results.

    Args:
        papers: List of classified papers
        cfg: Hydra configuration

    Returns:
        List of relevant papers
    """
    use_vllm = cfg.classification.get('use_vllm', True)
    use_openrouter = cfg.classification.get('use_openrouter', True)

    relevant = []

    for paper in papers:
        is_relevant = False

        if use_vllm and use_openrouter:
            # Both classifiers: require both to agree
            is_relevant = paper.get('vllm_relevant', False) and paper.get('openrouter_relevant', False)
        elif use_vllm:
            is_relevant = paper.get('vllm_relevant', False)
        elif use_openrouter:
            is_relevant = paper.get('openrouter_relevant', False)

        if is_relevant:
            relevant.append(paper)

    return relevant


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main function using Hydra config."""

    # Load classified papers
    input_file = f"{cfg.classification.output_prefix}_papers.json"

    if not os.path.exists(input_file):
        print(f"Error: Classified papers file not found: {input_file}")
        print("Please run classify_papers.py first.")
        return

    with open(input_file, 'r') as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} classified papers from {input_file}")

    # Filter relevant papers
    relevant_papers = get_relevant_papers(papers, cfg)

    print(f"\n{'='*80}")
    print(f"DOWNLOADING PDFS FOR {len(relevant_papers)} RELEVANT PAPERS")
    print(f"{'='*80}\n")

    if not relevant_papers:
        print("No relevant papers to download.")
        return

    # Create output directory
    output_dir = Path(cfg.download.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Process each relevant paper
    downloaded = 0
    skipped = 0
    failed = 0

    for i, paper in enumerate(relevant_papers, 1):
        title = paper.get('title', 'Unknown')[:60]

        print(f"[{i}/{len(relevant_papers)}] {title}...")

        # Try to get arXiv URL from openAccessPdf first, then fall back to url field
        arxiv_url = None
        arxiv_id = None

        # Check openAccessPdf field first (more reliable for arXiv papers)
        open_access = paper.get('openAccessPdf')
        if open_access and isinstance(open_access, dict):
            pdf_url = open_access.get('url', '')
            if 'arxiv.org' in pdf_url.lower():
                arxiv_url = pdf_url
                print(f"  Found arXiv URL in openAccessPdf")

        # Fall back to main url field if not found in openAccessPdf
        if not arxiv_url:
            main_url = paper.get('url', '')
            if 'arxiv.org' in main_url.lower():
                arxiv_url = main_url
                print(f"  Found arXiv URL in url field")

        # Extract arXiv ID from URL if we have one
        if arxiv_url:
            arxiv_id = extract_arxiv_id(arxiv_url)
            if arxiv_id:
                print(f"  arXiv ID: {arxiv_id}")

        # If we still don't have an arXiv ID, try searching by title
        if not arxiv_id:
            paper_title = paper.get('title', '')
            if paper_title:
                print(f"  No arXiv URL found, searching arXiv by title...")
                arxiv_id = search_arxiv_by_title(paper_title)

        # Final check - skip if we couldn't find the paper on arXiv
        if not arxiv_id:
            print(f"  ⊘ Not found on arXiv")
            skipped += 1
            continue

        # Download PDF
        result = download_arxiv_pdf(arxiv_id, output_dir, paper.get('title'))

        if result:
            downloaded += 1
        else:
            failed += 1

        # Rate limiting: sleep between downloads
        if i < len(relevant_papers):
            time.sleep(cfg.download.delay)

    # Summary
    print(f"\n{'='*80}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"Total relevant papers: {len(relevant_papers)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (not arXiv): {skipped}")
    print(f"Failed: {failed}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
