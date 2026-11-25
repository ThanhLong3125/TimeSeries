import os
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, Any

class ExperimentTracker:
    def __init__(self, audit_file: str = "lab4/logs/audit_trail.jsonl"):
        self.audit_file = audit_file
        os.makedirs(os.path.dirname(audit_file), exist_ok=True)

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        if not os.path.exists(file_path):
            return "FILE_NOT_FOUND"
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def log_event(self, stage: str, details: Dict[str, Any]):
        def json_converter(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": stage,
            "details": details
        }
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=json_converter) + "\n")
