# AWS Bedrock Setup for Deepseek

This project now uses **AWS Bedrock with Deepseek** for paper classification instead of local vLLM.

## Prerequisites

1. **AWS Account** with Bedrock access
2. **IAM credentials** with Bedrock permissions
3. **Deepseek model** enabled in your AWS region

## Setup Steps

### 1. Install AWS CLI (if not already installed)

```bash
# macOS
brew install awscli

# Or using pip
pip install awscli
```

### 2. Configure AWS Credentials

Choose one of these methods:

#### Option A: Interactive Configuration (Recommended)
```bash
aws configure
```

You'll be prompted for:
- AWS Access Key ID
- AWS Secret Access Key  
- Default region (e.g., `us-east-1`)
- Default output format (use `json`)

#### Option B: Environment Variables
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
```

#### Option C: AWS Credentials File
Create/edit `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

And `~/.aws/config`:
```ini
[default]
region = us-east-1
```

### 3. Verify AWS Configuration

```bash
# Test AWS credentials
aws sts get-caller-identity

# List available Bedrock models
aws bedrock list-foundation-models --region us-east-1
```

### 4. Request Bedrock Model Access

1. Go to AWS Console → Bedrock → Model access
2. Request access to the Deepseek model you want to use
3. Wait for approval (usually instant for most models)

### 5. Update Configuration

Edit `config.yaml` to set your preferred Bedrock configuration:

```yaml
bedrock:
  region: "us-east-1"  # Your AWS region
  model_id: "us.amazon.nova-micro-v1:0"  # Deepseek model ID
  max_tokens: 10
  temperature: 0.1
```

## Finding the Right Model ID

To find available Deepseek models in Bedrock:

```bash
# List all models (look for Deepseek)
aws bedrock list-foundation-models \
  --region us-east-1 \
  --query 'modelSummaries[*].[modelId, modelName]' \
  --output table
```

Common Deepseek model IDs in Bedrock:
- `deepseek-ai.deepseek-r1` (if available)
- `us.amazon.nova-micro-v1:0` (fallback: Amazon Nova Micro)
- Check AWS documentation for latest model IDs

## IAM Permissions Required

Your IAM user/role needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "*"
    }
  ]
}
```

## Usage

Once configured, simply run:

```bash
# Classify papers using Bedrock
python classify_papers.py

# The script will automatically use AWS Bedrock with your configured credentials
```

## Troubleshooting

### Error: "Could not connect to the endpoint URL"
- Check your AWS region in config.yaml matches where Bedrock is available
- Bedrock is available in: us-east-1, us-west-2, ap-southeast-1, eu-central-1, etc.

### Error: "UnrecognizedClientException"
- Verify your AWS credentials: `aws sts get-caller-identity`
- Reconfigure: `aws configure`

### Error: "AccessDeniedException"
- Request model access in AWS Console → Bedrock → Model access
- Verify IAM permissions include `bedrock:InvokeModel`

### Error: "ValidationException: The provided model identifier is invalid"
- Check model ID in config.yaml
- List available models: `aws bedrock list-foundation-models --region us-east-1`
- Update config.yaml with a valid model ID

## Cost Estimation

AWS Bedrock charges by:
- **Input tokens**: ~$0.003 per 1K tokens
- **Output tokens**: ~$0.006 per 1K tokens

For 1000 papers with ~200 token abstracts:
- Estimated cost: ~$2-5 USD
- Much cheaper than OpenRouter/OpenAI for bulk processing

## Switching Back to OpenRouter (Optional)

If you prefer OpenRouter, you can still use it:

1. Create a new classifier in `classify_papers.py`
2. Import `OpenRouterClassifier` instead of `BedrockClassifier`
3. Set `OPENROUTER_API_KEY` environment variable

The OpenRouter classifier is still available in the codebase.

