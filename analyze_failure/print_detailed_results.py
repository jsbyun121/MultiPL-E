#!/usr/bin/env python3
"""
Print detailed stdout/stderr results for each problem with index annotations.
This script reads the result files and prints the execution details for each trial.
"""

import json
import gzip
import os
from pathlib import Path
import re
import sys


def load_json_gz(file_path: str) -> dict:
    """Load and parse a gzipped JSON file."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_problem_number(filename: str) -> str:
    """Extract the problem number from filename (e.g., 'HumanEval_42_incr_list.results.json.gz' -> '42')."""
    match = re.search(r'HumanEval_(\d+)_', filename)
    if match:
        return match.group(1)
    return filename


def extract_problem_name(filename: str) -> str:
    """Extract the problem name from filename (e.g., 'HumanEval_42_incr_list.results.json.gz' -> 'incr_list')."""
    match = re.search(r'HumanEval_\d+_(.+?)\.results\.json\.gz', filename)
    if match:
        return match.group(1)
    return filename


def print_problem_results(data: dict, problem_number: str, problem_name: str, lang: str):
    """Print detailed results for a single problem."""
    if not data or 'results' not in data:
        print(f"No results found for problem {problem_number}")
        return
    
    results = data['results']
    if not results:
        print(f"Empty results for problem {problem_number}")
        return
    
    print("=" * 80)
    print(f"PROBLEM {problem_number}: {problem_name} ({lang.upper()})")
    print("=" * 80)
    
    for i, result in enumerate(results):
        print(f"\n--- Trial {i+1}/4 ---")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"Exit Code: {result.get('exit_code', 'Unknown')}")
        
        # Print stdout if available
        stdout = result.get('stdout', '')
        if stdout:
            print(f"\nSTDOUT:")
            print("-" * 40)
            print(stdout)
        else:
            print(f"\nSTDOUT: (empty)")
        
        # Print stderr if available
        stderr = result.get('stderr', '')
        if stderr:
            print(f"\nSTDERR:")
            print("-" * 40)
            print(stderr)
        else:
            print(f"\nSTDERR: (empty)")
        
        # Print execution time if available
        if 'execution_time' in result:
            print(f"\nExecution Time: {result['execution_time']}s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python print_detailed_results.py <language> [problem_numbers...]")
        print("Examples:")
        print("  python print_detailed_results.py rkt")
        print("  python print_detailed_results.py rkt 0 14 29")
        print("  python print_detailed_results.py jl 2 9 26")
        return
    
    language = sys.argv[1].lower()
    specific_problems = set(sys.argv[2:]) if len(sys.argv) > 2 else None
    
    base_path = "/home/junsoo/MultiPL-E/after_proc_openai_gpt-oss-20b_mt_4096/result"
    lang_path = os.path.join(base_path, language)
    
    if not os.path.exists(lang_path):
        print(f"Error: Language directory {lang_path} does not exist!")
        available_langs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        print(f"Available languages: {', '.join(available_langs)}")
        return
    
    # Get all JSON files for the language
    json_files = list(Path(lang_path).glob("*.results.json.gz"))
    json_files.sort(key=lambda x: int(extract_problem_number(x.name)))
    
    print(f"Processing {len(json_files)} files for language: {language.upper()}")
    
    if specific_problems:
        print(f"Filtering for problems: {', '.join(specific_problems)}")
    
    print()
    
    # Process each file
    for json_file in json_files:
        problem_number = extract_problem_number(json_file.name)
        
        # Skip if we're filtering for specific problems and this isn't one of them
        if specific_problems and problem_number not in specific_problems:
            continue
        
        problem_name = extract_problem_name(json_file.name)
        
        data = load_json_gz(str(json_file))
        if data is None:
            continue
        
        print_problem_results(data, problem_number, problem_name, language)
        print()  # Extra spacing between problems


if __name__ == "__main__":
    main()