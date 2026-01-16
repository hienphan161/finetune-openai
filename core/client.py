"""
OpenAI client setup and configuration.
"""

import os

from openai import OpenAI


def setup_openai_client(config: dict) -> OpenAI:
    """
    Initialize and return OpenAI client.
    
    Args:
        config: Configuration dictionary containing openai settings.
        
    Returns:
        Configured OpenAI client instance.
        
    Raises:
        ValueError: If API key is not found in config or environment.
    """
    openai_config = config.get("openai", {})
    
    api_key = openai_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = openai_config.get("base_url") or None
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set it in config.yaml or OPENAI_API_KEY environment variable."
        )
    
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    return OpenAI(**client_kwargs)

