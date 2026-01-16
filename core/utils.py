"""
Utility functions for configuration, file handling, and output management.
"""

import json
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing configuration settings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_results_dir() -> Path:
    """
    Ensure results directory exists and return its path.
    
    Returns:
        Path object for the results directory.
    """
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_output_filename(suffix: str, extension: str) -> str:
    """
    Generate output filename with timestamp.
    
    Args:
        suffix: Descriptive suffix for the file (e.g., "training_metrics").
        extension: File extension without dot (e.g., "csv").
        
    Returns:
        Formatted filename string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{suffix}_{timestamp}.{extension}"


def load_training_data(file_path: str) -> list[dict]:
    """
    Load training data from JSONL file.
    
    Args:
        file_path: Path to the JSONL training data file.
        
    Returns:
        List of dictionaries, each representing a training example.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

