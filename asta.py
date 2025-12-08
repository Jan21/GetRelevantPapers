import requests
import json
import textwrap
import hydra
from omegaconf import DictConfig

# --- Function to Fetch Papers ---
def search_asta_papers(keyword, fields, limit, api_key, asta_url):
    """
    Searches the ASTA Corpus using the search_papers_by_relevance tool.
    """
    # CORRECTED HEADERS: Accept both application/json and text/event-stream
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",  # Server requires both
        # Uncomment the line below to use your API key
        "x-api-key": api_key
    }

    # JSON-RPC 2.0 format required by the API
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_papers_by_relevance",
            "arguments": {
                "keyword": keyword,
                "fields": fields,
                "limit": limit
            }
        },
        "id": 1
    }

    print(f"Searching for papers on: '{keyword}'...")

    try:
        response = requests.post(asta_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse Server-Sent Events (SSE) format
        # Response format: "event: message\r\ndata: <JSON>\r\n\r\n"
        response_text = response.text.strip()

        # Extract JSON from SSE format
        if response_text.startswith('event:'):
            # Split by lines and find the data line
            lines = response_text.split('\r\n')
            for line in lines:
                if line.startswith('data: '):
                    json_str = line[6:]  # Remove 'data: ' prefix
                    data = json.loads(json_str)

                    # Extract the result from JSON-RPC response
                    if 'result' in data:
                        result = data['result']
                        # The actual papers are in result.structuredContent.result
                        if 'structuredContent' in result and 'result' in result['structuredContent']:
                            return result['structuredContent']['result']
                        # Fallback: return the content if it's a list of papers
                        elif 'content' in result:
                            return result['content']
                    return data
        else:
            # Fallback to regular JSON parsing
            data = response.json()
            return data

    except requests.exceptions.RequestException as e:
        print(f"\n--- API Request Failed ---")
        # Print status code if available for better debugging
        if response is not None and response.status_code:
            print(f"Status Code: {response.status_code}")
            # Attempt to print response content if it's an error message
            try:
                print(f"Response Body: {response.text}")
            except:
                pass
        
        print(f"An error occurred: {e}")
        return None

def remove_duplicates(papers):
    """
    Remove duplicate papers based on title or paper ID.

    Args:
        papers: List of paper dictionaries

    Returns:
        List of unique papers
    """
    seen_titles = set()
    seen_ids = set()
    unique_papers = []

    for paper in papers:
        # Use paper ID if available, otherwise use title
        paper_id = paper.get('paperId') or paper.get('id')
        title = paper.get('title', '').strip().lower()

        # Check if we've seen this paper before
        if paper_id and paper_id in seen_ids:
            continue
        if not paper_id and title in seen_titles:
            continue

        # Add to unique list
        unique_papers.append(paper)
        if paper_id:
            seen_ids.add(paper_id)
        if title:
            seen_titles.add(title)

    return unique_papers


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main function using Hydra config."""

    # Handle both single query (backward compatibility) and multiple queries
    queries = cfg.asta.get('search_queries', [cfg.asta.get('search_keyword', '')])
    if isinstance(queries, str):
        queries = [queries]

    all_results = []

    print(f"\n{'='*80}")
    print(f"SEARCHING FOR PAPERS WITH {len(queries)} QUERIES")
    print(f"{'='*80}\n")

    # Execute search for each query
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}] '{query}'")
        print("-" * 80)

        results = search_asta_papers(
            keyword=query,
            fields=cfg.asta.fields,
            limit=cfg.asta.limit,
            api_key=cfg.asta.api_key,
            asta_url=cfg.asta.url
        )

        if results and isinstance(results, list):
            print(f"✅ Retrieved {len(results)} papers for this query")
            all_results.extend(results)
        elif results is not None:
            print("⚠️ Unexpected response format:")
            print(json.dumps(results, indent=2)[:200])
        else:
            print("❌ Failed to retrieve papers for this query")

    if not all_results:
        print("\n❌ No papers retrieved from any query.")
        return

    print(f"\n{'='*80}")
    print(f"PROCESSING RESULTS")
    print(f"{'='*80}")
    print(f"Total papers retrieved: {len(all_results)}")

    # Remove duplicates
    unique_results = remove_duplicates(all_results)
    duplicates_removed = len(all_results) - len(unique_results)
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Unique papers: {len(unique_results)}")

    # Save results to JSON file
    output_file = cfg.asta.output
    with open(output_file, 'w') as f:
        json.dump(unique_results, f, indent=2)
    print(f"\n💾 Saved {len(unique_results)} unique papers to {output_file}")

    # Display papers
    print(f"\n{'='*80}")
    print(f"PAPER PREVIEW (showing first 10)")
    print(f"{'='*80}")

    for i, paper in enumerate(unique_results[:10], 1):
        print(f"\n[{i}] {paper.get('title', 'No Title')}")
        print(f"    Year: {paper.get('year', 'N/A')}")

        authors = paper.get('authors', [])
        if authors:
            author_names = ', '.join([a.get('name', 'Unknown') for a in authors[:3]])
            if len(authors) > 3:
                author_names += f" et al. ({len(authors)} total)"
            print(f"    Authors: {author_names}")

        if paper.get('abstract'):
            abstract = textwrap.fill(paper['abstract'][:200] + '...', width=80)
            print(f"    Abstract: {abstract}")

        if paper.get('url'):
            print(f"    URL: {paper['url']}")

        print("-" * 80)

    if len(unique_results) > 10:
        print(f"\n... and {len(unique_results) - 10} more papers")


if __name__ == "__main__":
    main()