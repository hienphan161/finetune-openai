#!/usr/bin/env python3
"""
OpenAI GPT Fine-tuning CLI

Command-line interface for fine-tuning GPT models using OpenAI's API.
"""

import argparse
import sys

from core import (
    setup_openai_client,
    load_config,
    load_training_data,
    validate_data_format,
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


def cmd_validate(args, config: dict) -> None:
    """Validate training data format."""
    training_file = args.file or config.get("fine_tuning", {}).get("training_file")
    
    if not training_file:
        print("Error: No training file specified.")
        sys.exit(1)
    
    print(f"Validating data format: {training_file}")
    data = load_training_data(training_file)
    errors = validate_data_format(data)
    
    if errors:
        print("\nFound errors:")
        for k, v in errors.items():
            print(f"  {k}: {v}")
        sys.exit(1)
    else:
        print(f"\nNo errors found. {len(data)} examples validated successfully.")


def cmd_upload(args, config: dict) -> None:
    """Upload training file to OpenAI."""
    client = setup_openai_client(config)
    training_file = args.file or config.get("fine_tuning", {}).get("training_file")
    
    if not training_file:
        print("Error: No training file specified.")
        sys.exit(1)
    
    file_id = upload_training_file(client, training_file)
    print(f"\nUse this file ID in your fine-tuning job: {file_id}")


def cmd_create(args, config: dict) -> None:
    """Create a fine-tuning job."""
    client = setup_openai_client(config)
    
    if not args.training_file_id:
        print("Error: --training-file-id is required.")
        sys.exit(1)
    
    job_id = create_fine_tuning_job(
        client,
        args.training_file_id,
        config,
        args.validation_file_id,
    )
    
    if args.wait:
        status = wait_for_job_completion(client, job_id)
        print(f"\nJob completed with status: {status['status']}")
        
        if status["fine_tuned_model"]:
            print(f"Fine-tuned model: {status['fine_tuned_model']}")
        
        if status["result_files"]:
            for file_id in status["result_files"]:
                download_result_file(client, file_id)


def cmd_status(args, config: dict) -> None:
    """Get status of a fine-tuning job."""
    client = setup_openai_client(config)
    
    if not args.job_id:
        print("Error: --job-id is required.")
        sys.exit(1)
    
    status = get_job_status(client, args.job_id)
    
    print("\nJob Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")


def cmd_list_files(args, config: dict) -> None:
    """List uploaded files."""
    client = setup_openai_client(config)
    files = list_files(client, args.limit)
    
    print(f"\nFiles (showing up to {args.limit}):")
    for i, f in enumerate(files):
        print(f"  {i+1}. {f['id']} - {f['filename']} ({f['status']})")


def cmd_list_jobs(args, config: dict) -> None:
    """List fine-tuning jobs."""
    client = setup_openai_client(config)
    jobs = list_jobs(client, args.limit)
    
    print(f"\nFine-tuning Jobs (showing up to {args.limit}):")
    for i, job in enumerate(jobs):
        model_info = job['fine_tuned_model'] or 'pending'
        print(f"  {i+1}. {job['id']} - {job['status']} - {model_info}")


def cmd_chat(args, config: dict) -> None:
    """Chat with a fine-tuned model."""
    client = setup_openai_client(config)
    
    if not args.model:
        print("Error: --model is required.")
        sys.exit(1)
    
    response = chat_with_model(
        client,
        args.model,
        args.message,
        args.system or "You are a helpful assistant.",
    )
    
    print(f"\nResponse:\n{response}")


def cmd_compare(args, config: dict) -> None:
    """Compare responses between base model and fine-tuned model."""
    client = setup_openai_client(config)
    
    # Get base model from args or config
    base_model = args.base_model or config.get("fine_tuning", {}).get("model", "gpt-3.5-turbo")
    
    if not args.finetuned_model:
        print("Error: --finetuned-model is required.")
        sys.exit(1)
    
    system_message = args.system or "You are a helpful assistant."
    
    comparison = compare_models(
        client,
        base_model,
        args.finetuned_model,
        args.message,
        system_message,
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("PROMPT")
    print("=" * 60)
    print(f"System: {comparison['system_message']}")
    print(f"User: {comparison['prompt']}")
    
    print("\n" + "=" * 60)
    print(f"BASE MODEL: {comparison['base_model']}")
    print("=" * 60)
    print(comparison['base_response'])
    
    print("\n" + "=" * 60)
    print(f"FINE-TUNED MODEL: {comparison['finetuned_model']}")
    print("=" * 60)
    print(comparison['finetuned_response'])


def cmd_run(args, config: dict) -> None:
    """Run the complete fine-tuning workflow."""
    client = setup_openai_client(config)
    training_file = args.file or config.get("fine_tuning", {}).get("training_file")
    
    if not training_file:
        print("Error: No training file specified.")
        sys.exit(1)
    
    # Step 1: Validate data
    print("\n=== Step 1: Validating training data ===")
    data = load_training_data(training_file)
    errors = validate_data_format(data)
    
    if errors:
        print("Found errors in training data:")
        for k, v in errors.items():
            print(f"  {k}: {v}")
        sys.exit(1)
    print(f"Validation passed. {len(data)} examples found.")
    
    # Step 2: Upload file
    print("\n=== Step 2: Uploading training file ===")
    file_id = upload_training_file(client, training_file)
    
    # Step 3: Create fine-tuning job
    print("\n=== Step 3: Creating fine-tuning job ===")
    job_id = create_fine_tuning_job(client, file_id, config)
    
    # Step 4: Wait for completion
    print("\n=== Step 4: Waiting for job completion ===")
    status = wait_for_job_completion(client, job_id)
    
    print(f"\nJob completed with status: {status['status']}")
    
    if status["status"] == "succeeded":
        print(f"\n✓ Fine-tuned model ready: {status['fine_tuned_model']}")
        
        # Download result files
        if status["result_files"]:
            print("\n=== Downloading result files ===")
            for result_file_id in status["result_files"]:
                download_result_file(client, result_file_id)
    else:
        print(f"\n✗ Job failed with error: {status.get('error')}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT models with OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate training data
  python finetune_openai.py validate --file fine-tuning-data.jsonl

  # Upload training file
  python finetune_openai.py upload --file fine-tuning-data.jsonl

  # Create fine-tuning job
  python finetune_openai.py create --training-file-id file-xxx --wait

  # Check job status
  python finetune_openai.py status --job-id ftjob-xxx

  # List files and jobs
  python finetune_openai.py list-files
  python finetune_openai.py list-jobs

  # Chat with fine-tuned model
  python finetune_openai.py chat --model ft:gpt-3.5-turbo:xxx --message "Hello!"

  # Compare base model vs fine-tuned model
  python finetune_openai.py compare --finetuned-model ft:gpt-3.5-turbo:xxx --message "Hello!"

  # Run complete workflow
  python finetune_openai.py run --file fine-tuning-data.jsonl
        """,
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate training data format")
    validate_parser.add_argument("--file", help="Path to training data file")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload training file to OpenAI")
    upload_parser.add_argument("--file", help="Path to training data file")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a fine-tuning job")
    create_parser.add_argument("--training-file-id", required=True, help="OpenAI file ID for training data")
    create_parser.add_argument("--validation-file-id", help="OpenAI file ID for validation data")
    create_parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get fine-tuning job status")
    status_parser.add_argument("--job-id", required=True, help="Fine-tuning job ID")
    
    # List files command
    list_files_parser = subparsers.add_parser("list-files", help="List uploaded files")
    list_files_parser.add_argument("--limit", type=int, default=20, help="Maximum files to list")
    
    # List jobs command
    list_jobs_parser = subparsers.add_parser("list-jobs", help="List fine-tuning jobs")
    list_jobs_parser.add_argument("--limit", type=int, default=20, help="Maximum jobs to list")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with a model")
    chat_parser.add_argument("--model", required=True, help="Model name/ID to use")
    chat_parser.add_argument("--message", required=True, help="User message")
    chat_parser.add_argument("--system", help="System message")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare base model vs fine-tuned model")
    compare_parser.add_argument("--base-model", help="Base model name (default: from config)")
    compare_parser.add_argument("--finetuned-model", required=True, help="Fine-tuned model ID")
    compare_parser.add_argument("--message", required=True, help="User message to send to both models")
    compare_parser.add_argument("--system", help="System message")
    
    # Run command (complete workflow)
    run_parser = subparsers.add_parser("run", help="Run complete fine-tuning workflow")
    run_parser.add_argument("--file", help="Path to training data file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Dispatch to command handler
    commands = {
        "validate": cmd_validate,
        "upload": cmd_upload,
        "create": cmd_create,
        "status": cmd_status,
        "list-files": cmd_list_files,
        "list-jobs": cmd_list_jobs,
        "chat": cmd_chat,
        "compare": cmd_compare,
        "run": cmd_run,
    }
    
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
