# GetRelevantPapers Terraform Infrastructure

This directory contains Terraform configurations for deploying GetRelevantPapers to AWS.

## Architecture

- **VPC**: Isolated network with public subnets across 2 AZs
- **ECS Fargate**: Serverless container orchestration
- **ECR**: Container registry for Docker images
- **Application Load Balancer**: Traffic distribution and health checks
- **Auto Scaling**: Automatic scaling based on CPU/Memory
- **CloudWatch**: Centralized logging and monitoring
- **IAM**: Least privilege access with Bedrock permissions

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Terraform** >= 1.0 installed
3. **AWS CLI** configured with credentials
4. **Docker** for building images

```bash
# Install Terraform
brew install terraform  # macOS
# or download from https://terraform.io

# Configure AWS CLI
aws configure
```

## Quick Start

### 1. Initialize Terraform

```bash
cd terraform
terraform init
```

### 2. Review Configuration

Edit `terraform.tfvars`:

```hcl
aws_region     = "us-east-1"
environment    = "prod"
instance_type  = "t3.medium"
min_capacity   = 1
max_capacity   = 3
enable_bedrock = true
```

### 3. Plan Deployment

```bash
terraform plan
```

### 4. Deploy Infrastructure

```bash
terraform apply
```

Type `yes` when prompted. Deployment takes ~10-15 minutes.

### 5. Build and Push Docker Image

```bash
# Get ECR repository URL
ECR_URL=$(terraform output -raw ecr_repository_url)

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URL

# Build image
cd ..
docker build -t getrelevantpapers .

# Tag and push
docker tag getrelevantpapers:latest $ECR_URL:latest
docker push $ECR_URL:latest
```

### 6. Update ECS Service

```bash
# Force new deployment
aws ecs update-service \
  --cluster $(terraform output -raw ecs_cluster_name) \
  --service $(terraform output -raw ecs_service_name) \
  --force-new-deployment
```

### 7. Access Application

```bash
# Get application URL
terraform output alb_url
```

Visit the URL in your browser (port 3444).

## Configuration

### Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | us-east-1 |
| `environment` | Environment name | prod |
| `container_port` | Application port | 3444 |
| `instance_type` | EC2 instance type | t3.medium |
| `min_capacity` | Min instances | 1 |
| `max_capacity` | Max instances | 3 |
| `enable_bedrock` | Enable AWS Bedrock | true |
| `allowed_cidr_blocks` | Access whitelist | 0.0.0.0/0 |

### Environments

Create environment-specific `.tfvars` files:

```bash
# Production
terraform apply -var-file=prod.tfvars

# Staging
terraform apply -var-file=staging.tfvars

# Development
terraform apply -var-file=dev.tfvars
```

Example `prod.tfvars`:

```hcl
environment      = "prod"
min_capacity     = 2
max_capacity     = 5
desired_capacity = 2
allowed_cidr_blocks = ["YOUR_IP/32"]  # Restrict access
```

## State Management

Terraform state is stored in S3 (configured in `main.tf`).

### Setup S3 Backend

```bash
# Create S3 bucket for state
aws s3 mb s3://getrelevantpapers-terraform-state --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket getrelevantpapers-terraform-state \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket getrelevantpapers-terraform-state \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

## Monitoring

### CloudWatch Logs

```bash
# View logs
aws logs tail /ecs/getrelevantpapers-prod --follow
```

### Metrics

Access CloudWatch metrics:
- ECS Service CPU/Memory utilization
- ALB request count and latency
- Container health checks

### Alarms

Set up CloudWatch alarms:

```bash
# High CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name getrelevantpapers-high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## Cost Optimization

### Estimated Monthly Costs

| Resource | Cost |
|----------|------|
| ECS Fargate (1 task) | ~$30 |
| Application Load Balancer | ~$16 |
| ECR Storage (10 GB) | ~$1 |
| CloudWatch Logs (5 GB) | ~$2.50 |
| Data Transfer | Variable |
| **Total** | **~$50-60/month** |

### Reduce Costs

1. **Use Spot instances** for dev/staging
2. **Schedule shutdown** for non-prod environments
3. **Reduce min_capacity** to 0 for dev
4. **Use Fargate Spot** pricing

## Scaling

### Manual Scaling

```bash
# Scale up
aws ecs update-service \
  --cluster getrelevantpapers-cluster-prod \
  --service getrelevantpapers-service-prod \
  --desired-count 3
```

### Auto Scaling

Configured automatically based on:
- **CPU**: Target 70%
- **Memory**: Target 80%

Adjust in `ecs.tf` if needed.

## Security

### IAM Permissions

The ECS task has permissions for:
- ✓ Bedrock API calls
- ✓ CloudWatch Logs
- ✓ ECR image pulls

### Network Security

- ALB: Public internet (configurable)
- ECS tasks: Private subnets with ALB access only
- Security groups: Least privilege rules

### Secrets Management

Use AWS Secrets Manager for sensitive data:

```hcl
# In ecs.tf, add to container_definitions:
secrets = [{
  name      = "API_KEY"
  valueFrom = "arn:aws:secretsmanager:region:account:secret:name"
}]
```

## Troubleshooting

### Service Won't Start

```bash
# Check service events
aws ecs describe-services \
  --cluster getrelevantpapers-cluster-prod \
  --services getrelevantpapers-service-prod \
  --query 'services[0].events'

# Check task logs
aws logs tail /ecs/getrelevantpapers-prod --follow
```

### Can't Access Application

1. Check security groups allow your IP
2. Verify ALB health checks are passing
3. Check DNS resolution

```bash
# Test health check
curl -v http://$(terraform output -raw alb_dns_name):3444/
```

### High Costs

```bash
# Check resource usage
aws ce get-cost-and-usage \
  --time-period Start=2025-12-01,End=2025-12-11 \
  --granularity DAILY \
  --metrics BlendedCost \
  --filter file://cost-filter.json
```

## Cleanup

### Destroy Infrastructure

```bash
# Preview what will be destroyed
terraform plan -destroy

# Destroy all resources
terraform destroy
```

**Warning**: This deletes all resources including logs and data!

### Partial Cleanup

```bash
# Scale down to 0
terraform apply -var="desired_capacity=0"

# Or delete specific resources
terraform destroy -target=aws_ecs_service.main
```

## CI/CD Integration

GitHub Actions automatically deploy on push to main.

See `.github/workflows/docker.yml` for the deployment pipeline.

## Support

For issues:
1. Check CloudWatch logs
2. Review Terraform plan output
3. Consult AWS ECS documentation
4. Open GitHub issue

## Additional Resources

- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)

