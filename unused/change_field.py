import os
import gzip
import json
import sys

def process_json_gz(file_path):
    try:
        # read
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        
        # raw_completions -> post_completions
        if "raw_completions" in data:
            data["post_completions"] = data.pop("raw_completions")
            modified = True
        
        # completions -> pre_completions
        if "completions" in data:
            data["pre_completions"] = data.pop("completions")
            modified = True
        
        # save if modified
        if modified:
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[UPDATED] {file_path}")
        else:
            print(f"[SKIP] {file_path} (no matching keys)")
    
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")

def process_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json.gz"):
                process_json_gz(os.path.join(dirpath, filename))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <root_directory>")
        sys.exit(1)
    
    root_directory = sys.argv[1]
    
    if not os.path.isdir(root_directory):
        print(f"Error: {root_directory} is not a valid directory.")
        sys.exit(1)
    
    process_directory(root_directory)
