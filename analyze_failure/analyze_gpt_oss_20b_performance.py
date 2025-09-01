#!/usr/bin/env python3
"""
Analyze gpt-oss-20b model performance data and categorize problems by trial accuracy.
This script reads the result files and categorizes problems based on how many trials
the model got correct (4, 3, 2, 1, or 0 out of 4 trials).
"""

import json
import gzip
import os
from pathlib import Path
from collections import defaultdict
import re


def load_json_gz(file_path: str) -> dict:
    """Load and parse a gzipped JSON file."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def count_successful_trials(data: dict) -> int:
    """Count the number of successful trials in a result file."""
    if not data or 'results' not in data:
        return 0
    
    results = data['results']
    if not results:
        return 0
    
    successful_count = 0
    
    for result in results:
        # Check if the trial was successful (status OK and exit_code 0)
        if result.get('status') == 'OK' and result.get('exit_code') == 0:
            successful_count += 1
    
    return successful_count


def extract_problem_number(filename: str) -> str:
    """Extract the problem number from filename (e.g., 'HumanEval_42_incr_list.results.json.gz' -> '42')."""
    match = re.search(r'HumanEval_(\d+)_', filename)
    if match:
        return match.group(1)
    return filename


def main():
    base_path = "/home/junsoo/MultiPL-E/after_proc_openai_gpt-oss-20b_mt_4096/result"
    
    # Get all available languages
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist!")
        return
    
    languages = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    languages.sort()
    
    print(f"Found languages: {languages}")
    print()
    
    # Dictionary to store results by number of successful trials
    # Format: {num_successful: {lang: set of problem numbers}}
    results_by_trials = {i: defaultdict(set) for i in range(5)}  # 0, 1, 2, 3, 4 successful trials
    
    # Process each language
    for lang in languages:
        lang_path = os.path.join(base_path, lang)
        json_files = Path(lang_path).glob("*.results.json.gz")
        
        for json_file in json_files:
            data = load_json_gz(str(json_file))
            if data is None:
                continue
                
            successful_trials = count_successful_trials(data)
            problem_number = extract_problem_number(json_file.name)
            
            # Store the result
            results_by_trials[successful_trials][lang].add(problem_number)
    
    # Print results for each trial count
    total_problems_per_category = {}
    
    for num_trials in range(5):
        print(f"=" * 60)
        print(f"PROBLEMS WITH {num_trials} OUT OF 4 TRIALS CORRECT:")
        print(f"=" * 60)
        
        category_total = 0
        
        for lang in languages:
            problems = sorted(results_by_trials[num_trials][lang], key=int)
            if problems:
                print(f"\n{lang.upper()} ({len(problems)} problems):")
                print("-" * 40)
                
                # Print problem numbers in rows of 10 for readability
                for i in range(0, len(problems), 10):
                    row = problems[i:i+10]
                    print(" ".join(f"{num:>3}" for num in row))
                
                category_total += len(problems)
        
        total_problems_per_category[num_trials] = category_total
        print(f"\nTotal problems with {num_trials} trials correct: {category_total}")
        print()
    
    # Summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS:")
    print("=" * 60)
    
    for num_trials in range(5):
        count = total_problems_per_category[num_trials]
        print(f"Problems with {num_trials}/4 trials correct: {count}")
    
    print(f"\nTotal problems analyzed: {sum(total_problems_per_category.values())}")
    
    # Calculate accuracy statistics
    print(f"\nAccuracy Distribution:")
    total_problems = sum(total_problems_per_category.values())
    if total_problems > 0:
        for num_trials in range(5):
            count = total_problems_per_category[num_trials]
            percentage = (count / total_problems) * 100
            accuracy = (num_trials / 4) * 100
            print(f"  {accuracy:5.1f}% accuracy ({num_trials}/4): {count:3d} problems ({percentage:5.1f}%)")


if __name__ == "__main__":
    main()