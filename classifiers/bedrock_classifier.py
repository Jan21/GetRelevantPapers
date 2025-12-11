"""
Paper classification using AWS Bedrock with Deepseek.
"""

from typing import Dict
from omegaconf import DictConfig
import boto3
import json
from .base_classifier import BaseClassifier


class BedrockClassifier(BaseClassifier):
    """Classifier for papers using AWS Bedrock with Deepseek."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize classifier with Hydra config.

        Args:
            cfg: Hydra configuration object
        """
        super().__init__(cfg)

        # Bedrock-specific configuration
        self.region = cfg.bedrock.get('region', 'us-east-1')
        self.model_id = cfg.bedrock.model_id
        self.max_tokens = cfg.bedrock.max_tokens
        self.temperature = cfg.bedrock.temperature
        
        # Initialize Bedrock client
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region
        )

    def _call_api(self, prompt: str) -> str:
        """
        Call AWS Bedrock API to get classification.

        Args:
            prompt: The formatted prompt

        Returns:
            Raw response text from the API
        """
        # Prepare the request body for Deepseek on Bedrock
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0
        }

        # Call Bedrock API
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )

        # Parse the response
        response_body = json.loads(response['body'].read())
        
        # Extract the generated text (format may vary by model)
        if 'content' in response_body:
            # Standard format
            generated = response_body['content'][0]['text']
        elif 'completion' in response_body:
            # Alternative format
            generated = response_body['completion']
        else:
            # Try to find the response in the output
            generated = str(response_body)

        return generated.strip()

