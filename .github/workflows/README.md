# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the GetRelevantPapers project.

## Workflows Overview

### 🔄 CI Pipeline (`ci.yml`)
**Triggers:** Push, Pull Request, Manual  
**Purpose:** Main continuous integration pipeline

**Jobs:**
- **Test** - Run tests across Python 3.8-3.11
- **Lint** - Code quality checks (Black, Flake8, Pylint)
- **Security** - Security scanning (Safety, Bandit)
- **Dependency Check** - Validate dependencies
- **Build Check** - Test imports and project structure

### 📚 Documentation (`deploy.yml`)
**Triggers:** Push to main/master, Manual  
**Purpose:** Build and deploy documentation to GitHub Pages

**Features:**
- Automatic API documentation generation
- Deploys markdown files to GitHub Pages
- Available at your repository's GitHub Pages URL

### 🔒 Security Updates (`dependency-update.yml`)
**Triggers:** Weekly (Mondays 9 AM UTC), Manual  
**Purpose:** Check for outdated and vulnerable dependencies

**Features:**
- Automated dependency auditing
- Creates issues for vulnerabilities
- Weekly security reports

### 🚀 Performance (`benchmark.yml`)
**Triggers:** Push (Python files), Pull Request, Manual  
**Purpose:** Monitor performance and memory usage

**Tests:**
- Analyzer initialization speed
- VectorStore performance
- Memory profiling
- Uploads benchmark artifacts

### 🐳 Docker (`docker.yml`)
**Triggers:** Push, Tags, Pull Request, Manual  
**Purpose:** Build and publish Docker images

**Features:**
- Multi-platform builds
- Automatic tagging
- Push to GitHub Container Registry
- Image testing

### 📊 Coverage (`coverage.yml`)
**Triggers:** Push, Pull Request, Manual  
**Purpose:** Generate code coverage reports

**Features:**
- Pytest with coverage
- HTML and XML reports
- PR comments with coverage stats
- Artifact uploads

### 🎉 Release (`release.yml`)
**Triggers:** Version tags (v*), Manual  
**Purpose:** Create GitHub releases

**Features:**
- Automatic changelog generation
- Source and minimal distribution archives
- Release notes compilation
- Asset uploads

### ✨ Auto Format (`format.yml`)
**Triggers:** Pull Request, Manual  
**Purpose:** Automatically format code

**Features:**
- Black formatter
- isort for imports
- Auto-commit formatting changes
- PR comments

### 🌙 Nightly Build (`nightly.yml`)
**Triggers:** Daily (2 AM UTC), Manual  
**Purpose:** Comprehensive integration testing

**Tests:**
- Full pipeline integration
- All module tests
- End-to-end workflows
- Failure notifications

## Status Badges

Add these to your README.md:

```markdown
![CI Pipeline](https://github.com/YOUR_USERNAME/GetRelevantPapers/workflows/CI%20Pipeline/badge.svg)
![Docker Build](https://github.com/YOUR_USERNAME/GetRelevantPapers/workflows/Docker%20Build%20%26%20Publish/badge.svg)
![Coverage](https://github.com/YOUR_USERNAME/GetRelevantPapers/workflows/Code%20Coverage/badge.svg)
```

## Required Secrets

No secrets are required for basic functionality. Optional secrets:

- `GITHUB_TOKEN` - Automatically provided by GitHub
- AWS credentials (if using Bedrock): Set as repository secrets
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION`

## Manual Triggers

All workflows can be manually triggered via:
1. Go to **Actions** tab in your repository
2. Select the workflow
3. Click **Run workflow**
4. Choose branch and parameters

## Customization

### Changing Python Versions
Edit `ci.yml`:
```yaml
matrix:
  python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

### Adjusting Schedule
Edit cron expressions:
```yaml
schedule:
  - cron: '0 2 * * *'  # Daily at 2 AM UTC
```

### Modifying Docker Registry
Edit `docker.yml`:
```yaml
env:
  REGISTRY: ghcr.io  # or docker.io, quay.io, etc.
```

## Troubleshooting

### Tests Failing
1. Check if dependencies need updating
2. Review test logs in Actions tab
3. Run tests locally: `pytest --verbose`

### Docker Build Issues
1. Verify Dockerfile exists
2. Check build context
3. Ensure all dependencies are in requirements.txt

### Coverage Not Generated
1. Ensure pytest-cov is installed
2. Check test discovery
3. Verify coverage configuration

## Best Practices

1. **Always run CI locally first**
   ```bash
   black .
   flake8 .
   pytest
   ```

2. **Keep workflows fast** - Use caching and parallel jobs

3. **Monitor workflow usage** - GitHub has minutes limits

4. **Update regularly** - Keep actions versions current

5. **Use continue-on-error carefully** - Only for non-critical checks

## Support

For issues with workflows:
1. Check workflow logs in Actions tab
2. Review this documentation
3. Open an issue with workflow name and error details

---

**Note:** These workflows are designed to work out-of-the-box but can be customized for your specific needs.

