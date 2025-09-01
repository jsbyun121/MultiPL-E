#!/usr/bin/env python3
import os
import sys
import re

def find_missing_problems(directory):
    """Find missing problem indices from 0 to 163 in the given directory."""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory")
        return
    
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Extract problem indices from filenames
    found_indices = set()
    
    for filename in files:
        # Look for patterns like "HumanEval_0", "HumanEval_123", etc.
        match = re.search(r'HumanEval_(\d+)', filename)
        if match:
            index = int(match.group(1))
            found_indices.add(index)
    
    # Find missing indices from 0 to 163
    expected_indices = set(range(164))  # 0 to 163 inclusive
    missing_indices = expected_indices - found_indices
    
    if missing_indices:
        missing_sorted = sorted(missing_indices)
        print(f"Missing problem indices: {missing_sorted}")
        print(f"Total missing: {len(missing_sorted)}")
    else:
        print("No missing problems found. All indices from 0 to 163 are present.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python find_missing_problems.py <directory>")
        print("Example: python find_missing_problems.py /home/junsoo/MultiPL-E/after_proc_openai_gpt-oss-20b_mt_4096/result/rkt")
        sys.exit(1)
    
    directory = sys.argv[1]
    find_missing_problems(directory)

if __name__ == "__main__":
    main()