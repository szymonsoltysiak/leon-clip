import wandb
import json
import pandas as pd

# Initialize the W&B API
api = wandb.Api()

# Configuration
entity = "szymon-soltysiak8-self " 
project = "geo-text-triple-encoder"
target_group = "Round2" # <--- Set your group name here

# Fetch runs filtered by group
# The filter "group" corresponds to the group name set in wandb.init(group="...")
runs = api.runs(f"{entity}/{project}", filters={"group": target_group})

extracted_data = []

print(f"Fetching runs for group: {target_group}...")

for run in runs:
    # Create a dictionary starting with basic run info
    run_dict = {
        "run_name": run.name,
        "run_id": run.id,
        "state": run.state,
        "group": run.group # Optional: keep track of the group name in the CSV
    }
    
    # run.summary contains the final values for all logged metrics
    run_dict.update(run.summary._json_dict)
    
    extracted_data.append(run_dict)

if not extracted_data:
    print(f"No runs found for group '{target_group}'. Check the name and try again.")
else:
    # Save as JSON
    with open("final_metrics.json", "w") as f:
        json.dump(extracted_data, f, indent=4)
    print(f"Saved {len(extracted_data)} runs to final_metrics.json")

    # Save as CSV
    df = pd.DataFrame(extracted_data)
    df.to_csv("final_metrics.csv", index=False)
    print(f"Saved {len(extracted_data)} runs to final_metrics.csv")