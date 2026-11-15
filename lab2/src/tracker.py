import json
import uuid
import hashlib
from datetime import datetime
import os
from .logger import logger

def calculate_file_md5(filepath):
    """Tính toán mã MD5 hash của file để đảm bảo tính toàn vẹn dữ liệu."""
    if not os.path.exists(filepath):
        return None
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def record_audit_entry(config, metrics, model_path, log_path="lab2/logs/audit_trail.jsonl", execution_time_sec=None):
    """Ghi nhận một bản ghi truy vết bất biến (Audit Trail) vào tệp JSON Lines."""
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    data_hash = calculate_file_md5(config["data_path"])
    
    entry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "experiment_name": config.get("experiment_name", "unnamed"),
        "dataset_name": config.get("dataset_name", "unnamed"),
        "data_path": config["data_path"],
        "data_version_md5": data_hash,
        "model_type": config["model"]["type"],
        "order": config["model"]["order"],
        "seasonal_order": config["model"].get("seasonal_order"),
        "train_ratio": config.get("train_ratio", 0.8),
        "execution_time_sec": round(execution_time_sec, 3) if execution_time_sec else None,
        "metrics": metrics,
        "model_artifact": model_path
    }
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
    logger.info(f"[AUDIT TRAIL] Recorded immutable run entry | run_id={run_id} | data_md5={data_hash}")
    return run_id
