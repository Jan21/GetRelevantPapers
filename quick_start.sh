#!/bin/bash
# Quick Start Script for AWS Bedrock Paper Classification
# This script helps you verify your setup before running classification

set -e

echo "=================================================="
echo "AWS Bedrock Paper Classification - Quick Start"
echo "=================================================="
echo ""

# Check Python
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi
python_version=$(python3 --version)
echo "✅ $python_version"
echo ""

# Check pip packages
echo "[2/6] Checking required packages..."
missing_packages=()

python3 -c "import boto3" 2>/dev/null || missing_packages+=("boto3")
python3 -c "import hydra" 2>/dev/null || missing_packages+=("hydra-core")
python3 -c "import omegaconf" 2>/dev/null || missing_packages+=("omegaconf")
python3 -c "import requests" 2>/dev/null || missing_packages+=("requests")

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "❌ Missing packages: ${missing_packages[*]}"
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi
echo "✅ All required packages installed"
echo ""

# Check AWS CLI
echo "[3/6] Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    echo "⚠️  AWS CLI not found (optional but recommended)"
    echo "   Install: brew install awscli (macOS) or pip install awscli"
else
    aws_version=$(aws --version 2>&1)
    echo "✅ $aws_version"
fi
echo ""

# Check AWS credentials
echo "[4/6] Checking AWS credentials..."
if python3 -c "import boto3; boto3.client('sts').get_caller_identity()" 2>/dev/null; then
    identity=$(aws sts get-caller-identity 2>/dev/null | grep -o '"Arn": "[^"]*"' | cut -d'"' -f4 || echo "Unknown")
    echo "✅ AWS credentials configured"
    echo "   Identity: $identity"
else
    echo "❌ AWS credentials not found or invalid"
    echo "   Run: aws configure"
    echo "   Or set: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    exit 1
fi
echo ""

# Check Bedrock access
echo "[5/6] Checking AWS Bedrock access..."
region=$(grep -A 5 "^bedrock:" config.yaml | grep "region:" | awk '{print $2}' | tr -d '"' || echo "us-east-1")
echo "   Region: $region"

if aws bedrock list-foundation-models --region "$region" &>/dev/null; then
    echo "✅ AWS Bedrock accessible in $region"
    
    # Check if Deepseek or Nova models are available
    model_count=$(aws bedrock list-foundation-models --region "$region" --query 'length(modelSummaries)' --output text 2>/dev/null || echo "0")
    echo "   Available models: $model_count"
else
    echo "❌ Cannot access AWS Bedrock in $region"
    echo "   Verify region and IAM permissions"
    exit 1
fi
echo ""

# Check for search results
echo "[6/6] Checking for papers to classify..."
if [ -f "search_results.json" ]; then
    paper_count=$(python3 -c "import json; print(len(json.load(open('search_results.json'))))" 2>/dev/null || echo "0")
    echo "✅ Found search_results.json with $paper_count papers"
else
    echo "⚠️  No search_results.json found"
    echo "   Run: python asta.py  OR  python semantic_scholar.py"
fi
echo ""

# Summary
echo "=================================================="
echo "Setup Status: ✅ READY"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. If you don't have papers yet:"
echo "     python asta.py                  # Search for papers"
echo ""
echo "  2. Classify papers with Bedrock:"
echo "     python classify_papers.py       # Run classification"
echo ""
echo "  3. View results:"
echo "     python visualize_papers.py      # Display results"
echo ""
echo "  4. Download PDFs:"
echo "     python download_papers.py       # Get relevant papers"
echo ""
echo "For more info, see BEDROCK_SETUP.md"
echo ""

