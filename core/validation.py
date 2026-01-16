"""
Data validation functions for fine-tuning datasets.
"""

from collections import defaultdict


def validate_data_format(data: list[dict]) -> dict:
    """
    Validate the format of training data for OpenAI fine-tuning.
    
    Checks for:
    - Correct data types
    - Required message fields (role, content)
    - Valid roles (system, user, assistant, function)
    - Presence of assistant messages
    
    Args:
        data: List of training examples loaded from JSONL.
        
    Returns:
        Dictionary of format errors found (empty if no errors).
    """
    format_errors = defaultdict(int)

    for ex in data:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call", "weight")
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    return dict(format_errors)

