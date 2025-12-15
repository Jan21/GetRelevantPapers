# Setup Instructions for Team Members

## AWS Bedrock Setup (Required for Analysis Feature)

To use the Bedrock analysis feature, you need AWS credentials with Bedrock access.

### What You Need

1. **AWS Access Key ID** - Provided by admin
2. **AWS Secret Access Key** - Provided by admin  
3. **AWS Region** - `us-east-1` (already in config.yaml)
4. **Bedrock Model Access** - Admin must enable model access

### Setup Methods

#### Method 1: Using AWS CLI (Recommended)

```bash
# Install AWS CLI if not installed
brew install awscli  # macOS
# or
pip install awscli

# Configure credentials
aws configure

# You'll be prompted for:
# AWS Access Key ID: [paste the key from admin]
# AWS Secret Access Key: [paste the secret from admin]
# Default region name: us-east-1
# Default output format: json
```

#### Method 2: Using Environment Variables

Create a `.env` file in the project root (it's gitignored):

```bash
# .env file
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_HERE
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY_HERE
AWS_REGION=us-east-1
```

Then load it before running the app:
```bash
export $(cat .env | xargs)
python src/ui/minimal_web_ui.py
```

#### Method 3: Direct Environment Variables

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"

python src/ui/minimal_web_ui.py
```

### Verify Setup

Test your AWS credentials:

```bash
# Check if credentials work
aws sts get-caller-identity

# List available Bedrock models
aws bedrock list-foundation-models --region us-east-1
```

### Required IAM Permissions

The AWS credentials need these permissions:

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

### Current Configuration

The project is configured to use:
- **Region**: `us-east-1` (in `config.yaml`)
- **Model**: `us.amazon.nova-micro-v1:0` (in `config.yaml`)

You don't need to change these unless instructed.

### Troubleshooting

**"Bedrock evaluator not available"**
- AWS credentials not configured
- Run `aws configure` or set environment variables

**"AccessDeniedException"**
- Credentials don't have Bedrock permissions
- Contact admin to enable model access in AWS Console

**"Could not connect to endpoint"**
- Wrong region configured
- Make sure you're using `us-east-1`

**Import errors**
- Missing dependencies
- Run: `pip install -r requirements.txt`

### Running the Web UI

Once AWS is configured:

```bash
python src/ui/minimal_web_ui.py
```

The web UI will open at http://localhost:3444 and you should see the "🚀 Analyze All (Bedrock - PARALLEL)" button.

### Security Note

**DO NOT commit your `.env` file or AWS credentials to git!** The `.gitignore` file already excludes `.env` files.

If credentials are compromised, contact the admin immediately to rotate them.

## Contact Admin For

- AWS Access Key ID
- AWS Secret Access Key
- Questions about Bedrock model access
- Permission issues

