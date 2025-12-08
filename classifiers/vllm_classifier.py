"""
Paper classification using VLLM server.
"""

import requests
from typing import Dict
from omegaconf import DictConfig
from .base_classifier import BaseClassifier


class VLLMClassifier(BaseClassifier):
    """Classifier for papers using VLLM server."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize classifier with Hydra config.

        Args:
            cfg: Hydra configuration object
        """
        super().__init__(cfg)

        # VLLM-specific configuration
        self.vllm_url = cfg.vllm.url.rstrip('/').rstrip('/v1')
        self.model_name = cfg.vllm.model_name
        self.max_tokens = cfg.vllm.max_tokens
        self.temperature = cfg.vllm.temperature
        self.timeout = cfg.vllm.timeout
        self.stop_tokens = cfg.vllm.stop_tokens
        self.session = requests.Session()

    def _call_api(self, prompt: str) -> str:
        """
        Call VLLM API to get classification.

        Args:
            prompt: The formatted prompt

        Returns:
            Raw response text from the API
        """
        # Format with Qwen3 chat template (ChatML format)
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Call vLLM OpenAI-compatible endpoint
        response = requests.post(
            f"{self.vllm_url}/v1/completions",
            json={
                "model": self.model_name,
                "prompt": full_prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stop": list(self.stop_tokens),
            },
            timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()

        # OpenAI-compatible API returns choices array
        if 'choices' in result and len(result['choices']) > 0:
            generated = result['choices'][0]['text']
        else:
            generated = ""

        # Clean up stop tokens
        text = generated.replace("<|im_end|>", "").strip()

        return text
