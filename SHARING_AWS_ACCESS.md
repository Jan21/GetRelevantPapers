# How to Share AWS Bedrock Access with Your Colleague

## Quick Answer: What to Share

Your colleague needs these **3 pieces of information** to use Bedrock analysis:

1. **AWS Access Key ID** - Example: `AKIAIOSFODNN7EXAMPLE`
2. **AWS Secret Access Key** - Example: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
3. **AWS Region** - Currently set to: `us-east-1` (already in config.yaml)

## ⚠️ Security Warning

**These credentials should be:**
- ✅ Sent via secure channel (Signal, 1Password, LastPass, etc.)
- ✅ Temporary if possible (use IAM roles with limited permissions)
- ❌ **NEVER** committed to git
- ❌ **NEVER** sent via email or Slack (unless encrypted)
- ❌ **NEVER** shared publicly

## Option 1: Share Your Credentials (Quick but Less Secure)

If you want your colleague to use **your** AWS account temporarily:

### Step 1: Get Your AWS Credentials

**If you configured with `aws configure`:**
```bash
# View your credentials
cat ~/.aws/credentials
```

You'll see something like:
```ini
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

**If you use environment variables:**
```bash
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
```

### Step 2: Send to Your Colleague Securely

Send them:
1. The AWS Access Key ID
2. The AWS Secret Access Key
3. Link to setup instructions: `docs/SETUP_FOR_COLLEAGUES.md`

### Step 3: They Configure Their Machine

Your colleague follows the instructions in `docs/SETUP_FOR_COLLEAGUES.md`

## Option 2: Create Separate IAM User (Recommended)

For better security, create a separate AWS IAM user for your colleague:

### Step 1: Create IAM User

1. Go to AWS Console → IAM → Users
2. Click "Add user"
3. Username: `colleague-name-bedrock`
4. Select "Access key - Programmatic access"
5. Click "Next: Permissions"

### Step 2: Attach Bedrock Permissions

Choose "Attach existing policies directly" and create custom policy:

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

Or attach AWS managed policy: `AmazonBedrockFullAccess`

### Step 3: Download Credentials

1. Complete user creation
2. **IMPORTANT**: Download the CSV with credentials (you can only see them once!)
3. Send the credentials securely to your colleague

### Step 4: Your Colleague Configures

They use the new credentials following `docs/SETUP_FOR_COLLEAGUES.md`

## What's Already Configured (No Need to Share)

These are already in the repository via `config.yaml`:

- ✅ AWS Region: `us-east-1`
- ✅ Bedrock Model: `us.amazon.nova-micro-v1:0`
- ✅ Model parameters (temperature, max_tokens, etc.)

Your colleague **doesn't** need to change these.

## Files to Share with Your Colleague

1. **This repository** (they should clone it)
2. **AWS credentials** (via secure channel)
3. **Setup instructions**: Point them to `docs/SETUP_FOR_COLLEAGUES.md`
4. **Example .env file**: `.env.example` shows the format

## Quick Setup Instructions for Colleague

```bash
# 1. Clone repository
git clone <your-repo-url>
cd GetRelevantPapers

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure AWS credentials (Method A: AWS CLI)
aws configure
# Paste the Access Key ID
# Paste the Secret Access Key
# Region: us-east-1
# Format: json

# OR Method B: Create .env file
cp .env.example .env
# Edit .env and add the credentials

# 4. Run the web UI
python ui/minimal_web_ui.py
```

The web UI will open at http://localhost:3444 and they should see the "🚀 Analyze All (Bedrock - PARALLEL)" button.

## Verify It's Working

Your colleague can test if AWS is configured correctly:

```bash
# Test AWS credentials
aws sts get-caller-identity

# Test Bedrock access
aws bedrock list-foundation-models --region us-east-1

# Run the web UI
python ui/minimal_web_ui.py
```

If everything works, the web UI console should show:
```
✓ Components initialized
  - Bedrock: ✓ Available
```

## Troubleshooting

**Colleague sees "Bedrock evaluator not available"**
- AWS credentials not configured
- Run `aws configure` or check `.env` file

**"AccessDeniedException" error**
- Credentials don't have Bedrock permissions
- Add `bedrock:InvokeModel` permission to IAM user

**"Could not connect to endpoint" error**
- Wrong region
- Make sure using `us-east-1` (or change in `config.yaml`)

**Bedrock button still doesn't appear**
- Check if `boto3` is installed: `pip install boto3`
- Check console output when starting web UI
- Look for import errors

## Security Best Practices

1. **Rotate credentials regularly** (every 90 days)
2. **Use IAM roles** when possible (especially on AWS infrastructure)
3. **Monitor usage** in AWS CloudWatch
4. **Revoke access** when colleague no longer needs it
5. **Use temporary credentials** for contractors/short-term access

## Contact Info

If your colleague has issues:
- Read `docs/SETUP_FOR_COLLEAGUES.md`
- Read `docs/BEDROCK_SETUP.md`
- Check AWS credentials configuration
- Verify IAM permissions in AWS Console

## Summary

**To enable Bedrock for your colleague, share:**
- AWS Access Key ID
- AWS Secret Access Key
- Point them to `docs/SETUP_FOR_COLLEAGUES.md`

**They need to:**
- Clone the repository
- Install requirements: `pip install -r requirements.txt`
- Configure AWS credentials (via `aws configure` or `.env` file)
- Run: `python ui/minimal_web_ui.py`

That's it! The Bedrock button should appear in the web UI.

