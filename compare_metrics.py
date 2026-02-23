import json
import sys
import os

new_metrics_file = "metrics.json"
old_metrics_file = "previous_metrics.json"

if not os.path.exists(old_metrics_file):
    print("No previous metrics found. First build.")
    sys.exit(0)

with open(new_metrics_file, "r") as f:
    new_metrics = json.load(f)

with open(old_metrics_file, "r") as f:
    old_metrics = json.load(f)

if new_metrics["accuracy"] < old_metrics["accuracy"]:
    print("Model performance decreased. Failing pipeline.")
    sys.exit(1)
else:
    print("Model improved or stayed same.")
    sys.exit(0)
