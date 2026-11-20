import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The pipeline modules pick their mode from environment variables at import time. Tests run the
# plain rules: no Smart Tagging, no stoppage model, and never the real match data.
for name in ("MTAH_TAGS", "MTAH_STOPMODEL", "MTAH_POSMODEL", "MTAH_VERSION"):
    os.environ.pop(name, None)
os.environ["MTAH_DATA_DIR"] = os.path.join(ROOT, "tests", "no-data")
os.environ["MTAH_WORK_DIR"] = os.path.join(ROOT, "tests", "no-work")

for folder in ("", "analysis", "possession", "dead_ball", "ball_recovery", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, folder))
