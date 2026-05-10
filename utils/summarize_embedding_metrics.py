import os
import json
import csv
from pathlib import Path

def generate_summary():
    # Base directory based on your tree output
    base_dir = "~/repo/leon-clip/outputs/embeddings_eval"
    base_path = Path(base_dir).expanduser()
    output_file = "alignment_metrics_summary.csv"
    
    # Define the exact header requested
    header = [
        "stage", "modality", "geo_corr_pearson", "geo_corr_spearman", 
        "effective_dim", "isotropy", "participation_ratio", 
        "uniformity0.5", "uniformity1.0", "uniformity2.0", 
        "uniformity5.0", "uniformity10.0"
    ]
    
    rows = []
    
    # Recursively find all json files in the directory
    for json_file in base_path.rglob("*.json"):
        # The directory structure is .../stage/modality/filename.json
        # We can extract stage and modality directly from the parent folders
        modality = json_file.parent.name
        stage = json_file.parent.parent.name
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Grab the root key dynamically (e.g., "text", "graph", "image")
                root_key = list(data.keys())[0]
                content = data[root_key]
                
                # Extract sub-dictionaries
                geo_corr = content.get('geo_corr', {})
                stats = content.get('stats', {})
                uniformity = content.get('uniformity', {})
                
                # Map the data to our row format
                row = [
                    stage,
                    modality,
                    geo_corr.get('pearson', 'N/A'),
                    geo_corr.get('spearman', 'N/A'),
                    stats.get('effective_dim', 'N/A'),
                    stats.get('isotropy', 'N/A'),
                    stats.get('participation_ratio', 'N/A'),
                    uniformity.get('0.5', 'N/A'),
                    uniformity.get('1.0', 'N/A'),
                    uniformity.get('2.0', 'N/A'),
                    uniformity.get('5.0', 'N/A'),
                    uniformity.get('10.0', 'N/A')
                ]
                rows.append(row)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # Sort rows for a nice, predictable output order (e.g., all 'post' together, then 'pre')
    rows.sort(key=lambda x: (x[0], x[1]))

    # Write the output to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        
    print(f"Successfully processed {len(rows)} files.")
    print(f"Summary saved to: {Path(output_file).resolve()}")

if __name__ == "__main__":
    generate_summary()