"""
Base class for paper classification.
"""

from abc import ABC, abstractmethod
from typing import Dict
from omegaconf import DictConfig


class BaseClassifier(ABC):
    """Abstract base class for paper classifiers."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize classifier with Hydra config.

        Args:
            cfg: Hydra configuration object
        """
        self.prompt_template = cfg.prompt.template
        self.user_description = cfg.classification.research_description

    def set_classification_criteria(self, description: str):
        """Set the user's description of what papers they're looking for."""
        self.user_description = description

    def _prepare_prompt(self, paper: Dict) -> str:
        """
        Prepare the prompt for classification.

        Args:
            paper: Paper dictionary with title and abstract

        Returns:
            Formatted prompt string
        """
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')

        # Create the user message from template
        user_message = self.prompt_template.format(
            research_description=self.user_description,
            title=title,
            abstract=" ".join(abstract.split(' '))
        )

        return user_message

    def _parse_response(self, response_text: str) -> bool:
        """
        Parse the classification response.

        Args:
            response_text: Raw response text from model

        Returns:
            True if relevant, False otherwise
        """
        text = response_text.strip().upper()

        if "YES" in text:
            return True
        elif "NO" in text:
            return False
        else:
            # If unclear response, default to False (skip)
            print(f"  ⚠ Unclear response '{text}' for paper, defaulting to NO")
            return False

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """
        Call the API to get classification.
        Must be implemented by subclasses.

        Args:
            prompt: The formatted prompt

        Returns:
            Raw response text from the API
        """
        pass

    def classify_paper(self, paper: Dict) -> bool:
        """
        Classify a paper based on its abstract.

        Args:
            paper: Paper dictionary with title and abstract

        Returns:
            True if paper is relevant, False otherwise
        """
        if not self.user_description:
            raise ValueError("Classification criteria not set. Call set_classification_criteria first.")

        abstract = paper.get('abstract', '')
        title = paper.get('title', '')

        # Handle papers without abstracts
        if not abstract or abstract.strip() == '':
            print(f"  ⊘ Skipping (no abstract): {title[:60]}...")
            return False

        try:
            # Prepare prompt
            prompt = self._prepare_prompt(paper)

            # Call API (implemented by subclass)
            response_text = self._call_api(prompt)

            # Parse response
            return self._parse_response(response_text)

        except Exception as e:
            print(f"  ⚠ Error classifying paper: {e}")
            return False
