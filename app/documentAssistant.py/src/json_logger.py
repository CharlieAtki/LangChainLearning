import json
import datetime
import os
from typing import Any, Dict


# Path to the JSONL log file
LOG_FILE = "logs.jsonl"


def log(message: str, **data: Dict[str, Any]):
    """
    Append a structured JSON log entry to logs.jsonl.

    Parameters:
    - message (str): Description of the log event
    - **data: Arbitrary key-value pairs to include in the log
    """
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "message": message,
        "data": data,
    }

    # Make sure the directory exists
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

    # Append as one line per JSON record
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
