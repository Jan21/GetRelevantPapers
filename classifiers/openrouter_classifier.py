"""
Paper classification using OpenRouter API.
"""

from typing import Dict
from omegaconf import DictConfig
from openai import OpenAI
from .base_classifier import BaseClassifier


class OpenRouterClassifier(BaseClassifier):
    """Classifier for papers using OpenRouter API."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize classifier with Hydra config.

        Args:
            cfg: Hydra configuration object
        """
        super().__init__(cfg)

        # OpenRouter-specific configuration
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=cfg.openrouter.api_key,
        )
        self.model_name = cfg.openrouter.model_name
        self.max_tokens = cfg.openrouter.max_tokens
        self.temperature = cfg.openrouter.temperature
        self.timeout = cfg.openrouter.timeout

    def _call_api(self, prompt: str) -> str:
        """
        Call OpenRouter API to get classification.

        Args:
            prompt: The formatted prompt

        Returns:
            Raw response text from the API
        """
        # Call OpenRouter API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

        # Extract the response text
        generated = response.choices[0].message.content.strip()

        return generated
