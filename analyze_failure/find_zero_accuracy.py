#!/usr/bin/env python3
"""
Find JSON.gz files that have 0% accuracy for each language.
"""

import json
import gzip
import os
from pathlib import Path
from collections import defaultdict


def load_json_gz(file_path: str) -> dict:
    """Load and parse a gzipped JSON file."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None


def calculate_accuracy(data: dict) -> float:
    """Calculate accuracy for a test file based on results."""
    if not data or 'results' not in data:
        return 0.0
    
    results = data['results']
    if not results:
        return 0.0
    
    successful_count = 0
    total_count = len(results)
    
    for result in results:
        if result.get('status') == 'OK' and result.get('exit_code') == 0:
            successful_count += 1
    
    return successful_count / total_count if total_count > 0 else 0.0


def main():
    base_path = "/home/junsoo/MultiPL-E/after_proc_openai_gpt-oss-20b_mt_4096/result"
    languages = ['jl', 'lua', 'ml', 'r', 'rkt']
    
    # Track zero accuracy files per language
    zero_accuracy_files = defaultdict(set)
    
    # Analyze each language
    for lang in languages:
        lang_path = os.path.join(base_path, lang)
        if not os.path.exists(lang_path):
            continue
            
        json_files = Path(lang_path).glob("*.json.gz")
        
        for json_file in json_files:
            data = load_json_gz(str(json_file))
            if data and calculate_accuracy(data) == 0.0:
                zero_accuracy_files[lang].add(json_file.name)
    
    # Print results for each language
    for lang in languages:
        if lang in zero_accuracy_files:
            sorted_files = sorted(zero_accuracy_files[lang])
            print(f"\n{lang.upper()} - Questions with 0% accuracy:")
            print("=" * 50)
            for i, filename in enumerate(sorted_files, 1):
                print(f"{i:3d}. {filename}")
            print(f"Total: {len(sorted_files)} files")
        else:
            print(f"\n{lang.upper()} - No files with 0% accuracy found")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY:")
    total_files_per_lang = {}
    for lang in languages:
        count = len(zero_accuracy_files[lang]) if lang in zero_accuracy_files else 0
        total_files_per_lang[lang] = count
        print(f"{lang.upper()}: {count} questions with 0% accuracy")
    
    # Find universal zero accuracy files
    if zero_accuracy_files:
        universal_zero_files = set(zero_accuracy_files[languages[0]])
        for lang in languages[1:]:
            universal_zero_files = universal_zero_files.intersection(zero_accuracy_files[lang])
        
        if universal_zero_files:
            print(f"\nQuestions with 0% accuracy across ALL languages: {len(universal_zero_files)}")
            for filename in sorted(universal_zero_files):
                print(f"  - {filename}")
        else:
            print(f"\nQuestions with 0% accuracy across ALL languages: 0")


if __name__ == "__main__":
    main()