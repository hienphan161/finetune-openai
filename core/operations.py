"""
Core operations for OpenAI fine-tuning: upload, job management, chat, and results.
"""

import base64
import time
from datetime import datetime
from typing import Optional

from openai import OpenAI

from core.utils import ensure_results_dir, get_output_filename


def upload_training_file(client: OpenAI, file_path: str) -> str:
    """
    Upload training file to OpenAI.
    
    Args:
        client: OpenAI client instance.
        file_path: Path to the training data file.
        
    Returns:
        File ID from OpenAI.
    """
    print(f"Uploading training file: {file_path}")
    
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    
    print(f"File uploaded successfully. File ID: {response.id}")
    return response.id


def create_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    config: dict,
    validation_file_id: Optional[str] = None,
) -> str:
    """
    Create a fine-tuning job.
    
    Args:
        client: OpenAI client instance.
        training_file_id: OpenAI file ID for training data.
        config: Configuration dictionary.
        validation_file_id: Optional OpenAI file ID for validation data.
        
    Returns:
        Fine-tuning job ID.
    """
    ft_config = config.get("fine_tuning", {})
    hyperparams = ft_config.get("hyperparameters", {})
    
    # Build hyperparameters dict, only including non-auto values
    hyperparameters = {}
    for param in ["n_epochs", "batch_size", "learning_rate_multiplier"]:
        value = hyperparams.get(param, "auto")
        if value != "auto":
            hyperparameters[param] = value
    
    job_kwargs = {
        "training_file": training_file_id,
        "model": ft_config.get("model", "gpt-3.5-turbo"),
    }
    
    if hyperparameters:
        job_kwargs["hyperparameters"] = hyperparameters
    
    if validation_file_id:
        job_kwargs["validation_file"] = validation_file_id
    
    suffix = ft_config.get("suffix")
    if suffix:
        job_kwargs["suffix"] = suffix
    
    print(f"Creating fine-tuning job with model: {job_kwargs['model']}")
    response = client.fine_tuning.jobs.create(**job_kwargs)
    
    print(f"Fine-tuning job created. Job ID: {response.id}")
    return response.id


def get_job_status(client: OpenAI, job_id: str) -> dict:
    """
    Get the status of a fine-tuning job.
    
    Args:
        client: OpenAI client instance.
        job_id: Fine-tuning job ID.
        
    Returns:
        Dictionary containing job status information.
    """
    job = client.fine_tuning.jobs.retrieve(job_id)
    return {
        "id": job.id,
        "status": job.status,
        "model": job.model,
        "fine_tuned_model": job.fine_tuned_model,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "result_files": job.result_files,
    }


def wait_for_job_completion(
    client: OpenAI, job_id: str, poll_interval: int = 30
) -> dict:
    """
    Wait for a fine-tuning job to complete, polling periodically.
    
    Args:
        client: OpenAI client instance.
        job_id: Fine-tuning job ID.
        poll_interval: Seconds between status checks.
        
    Returns:
        Final job status dictionary.
    """
    print(f"Waiting for job {job_id} to complete...")
    
    while True:
        status = get_job_status(client, job_id)
        current_status = status["status"]
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {current_status}")
        
        if current_status in ("succeeded", "failed", "cancelled"):
            return status
        
        time.sleep(poll_interval)


def list_files(client: OpenAI, limit: int = 20) -> list[dict]:
    """
    List files uploaded to OpenAI.
    
    Args:
        client: OpenAI client instance.
        limit: Maximum number of files to return.
        
    Returns:
        List of file information dictionaries.
    """
    files = []
    for f in client.files.list():
        files.append({
            "id": f.id,
            "filename": f.filename,
            "created_at": f.created_at,
            "purpose": f.purpose,
            "status": f.status,
        })
        if len(files) >= limit:
            break
    return files


def list_jobs(client: OpenAI, limit: int = 20) -> list[dict]:
    """
    List fine-tuning jobs.
    
    Args:
        client: OpenAI client instance.
        limit: Maximum number of jobs to return.
        
    Returns:
        List of job information dictionaries.
    """
    jobs = []
    for job in client.fine_tuning.jobs.list():
        jobs.append({
            "id": job.id,
            "model": job.model,
            "status": job.status,
            "fine_tuned_model": job.fine_tuned_model,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
        })
        if len(jobs) >= limit:
            break
    return jobs


def chat_with_model(
    client: OpenAI,
    model: str,
    user_message: str,
    system_message: str = "You are a helpful assistant.",
) -> str:
    """
    Send a chat message to the model and return the response.
    
    Args:
        client: OpenAI client instance.
        model: Model name or fine-tuned model ID.
        user_message: User's message content.
        system_message: System prompt for the conversation.
        
    Returns:
        Assistant's response content.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content


def compare_models(
    client: OpenAI,
    base_model: str,
    finetuned_model: str,
    user_message: str,
    system_message: str = "You are a helpful assistant.",
) -> dict:
    """
    Compare responses between base model and fine-tuned model.
    
    Args:
        client: OpenAI client instance.
        base_model: Base model name (e.g., gpt-3.5-turbo).
        finetuned_model: Fine-tuned model ID.
        user_message: User's message content.
        system_message: System prompt for the conversation.
        
    Returns:
        Dictionary containing both responses.
    """
    print(f"Sending prompt to both models...")
    print(f"  Base model: {base_model}")
    print(f"  Fine-tuned model: {finetuned_model}")
    
    base_response = chat_with_model(client, base_model, user_message, system_message)
    finetuned_response = chat_with_model(client, finetuned_model, user_message, system_message)
    
    return {
        "prompt": user_message,
        "system_message": system_message,
        "base_model": base_model,
        "base_response": base_response,
        "finetuned_model": finetuned_model,
        "finetuned_response": finetuned_response,
    }


def download_result_file(client: OpenAI, file_id: str) -> str:
    """
    Download and save the result file from a fine-tuning job.
    
    Args:
        client: OpenAI client instance.
        file_id: OpenAI file ID for the result file.
        
    Returns:
        Path to the saved file.
    """
    results_dir = ensure_results_dir()
    
    file_contents = client.files.content(file_id)
    decoded_bytes = base64.b64decode(file_contents.read())
    decoded_str = decoded_bytes.decode("utf-8")
    
    output_filename = get_output_filename("training_metrics", "csv")
    output_path = results_dir / output_filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(decoded_str)
    
    print(f"Result file saved to: {output_path}")
    return str(output_path)

