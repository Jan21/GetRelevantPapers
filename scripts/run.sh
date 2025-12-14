#!/bin/bash
# Example workflow: Search and classify papers using Hydra configuration

set -e  # Exit on error

echo "=================================="
echo "Search and Classify Workflow Example"
echo "=================================="
echo ""

echo "Step 1: Searching for papers..."
echo "Using configuration from config.yaml"
echo ""

python asta.py

echo ""
echo "=================================="
echo "Step 2: Classifying papers..."
echo "=================================="
echo ""

python classify_papers.py


