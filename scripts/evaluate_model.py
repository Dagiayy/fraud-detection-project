# scripts/evaluate_model.py
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import settings

def main():
    if settings.MODEL_METADATA_PATH.exists():
        with open(settings.MODEL_METADATA_PATH, "r") as f:
            meta = json.load(f)
            print(json.dumps(meta, indent=2))
    else:
        print(f"❌ Model metadata file not found at {settings.MODEL_METADATA_PATH}. Please run training first.")

if __name__ == "__main__":
    main()
