"""
Core module for OpenAI fine-tuning operations.
"""

from core.client import setup_openai_client
from core.utils import (
    load_config,
    ensure_results_dir,
    get_output_filename,
    load_training_data,
)
from core.validation import validate_data_format
from core.operations import (
    upload_training_file,
    create_fine_tuning_job,
    get_job_status,
    wait_for_job_completion,
    list_files,
    list_jobs,
    chat_with_model,
    compare_models,
    download_result_file,
)

__all__ = [
    # Client
    "setup_openai_client",
    # Utils
    "load_config",
    "ensure_results_dir",
    "get_output_filename",
    "load_training_data",
    # Validation
    "validate_data_format",
    # Operations
    "upload_training_file",
    "create_fine_tuning_job",
    "get_job_status",
    "wait_for_job_completion",
    "list_files",
    "list_jobs",
    "chat_with_model",
    "compare_models",
    "download_result_file",
]

