#!/usr/bin/env python3
"""
Find problems where one language performed better than another.
Sort by accuracy difference in descending order.
Usage: python find_language_accuracy_differences.py <base_dir> <lang1> <lang2>
Example: python find_language_accuracy_differences.py /path/to/results lua ml
"""

import json
import gzip
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_json_gz(file_path: str) -> Dict:
    """Load and parse a gzipped JSON file."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_accuracy(data: Dict) -> float:
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


def analyze_directory(directory_path: str, language: str) -> Dict[str, Tuple[float, Dict]]:
    """Analyze all JSON.gz files in a directory and return accuracy data by filename."""
    results = {}
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return results
    
    # Find all .json.gz files
    json_files = list(directory.glob("*.json.gz"))
    print(f"Found {len(json_files)} JSON.gz files for {language.upper()}")
    
    for json_file in json_files:
        data = load_json_gz(str(json_file))
        
        if data is not None:
            accuracy = calculate_accuracy(data)
            results[json_file.name] = (accuracy, data)
    
    return results


def find_differences(lang1_results: Dict[str, Tuple[float, Dict]], 
                    lang2_results: Dict[str, Tuple[float, Dict]], 
                    lang1: str, lang2: str) -> List[Tuple[str, float, float, float, Dict, Dict]]:
    """
    Find problems where lang1 performed better than lang2.
    Returns list of (filename, lang1_accuracy, lang2_accuracy, difference, lang1_data, lang2_data)
    sorted by difference descending.
    """
    differences = []
    
    # Find common files
    common_files = set(lang1_results.keys()) & set(lang2_results.keys())
    print(f"Found {len(common_files)} common files between {lang1.upper()} and {lang2.upper()}")
    
    for filename in common_files:
        lang1_accuracy, lang1_data = lang1_results[filename]
        lang2_accuracy, lang2_data = lang2_results[filename]
        
        # We want cases where lang1 did better than lang2
        if lang1_accuracy > lang2_accuracy:
            difference = lang1_accuracy - lang2_accuracy
            differences.append((filename, lang1_accuracy, lang2_accuracy, difference, lang1_data, lang2_data))
    
    # Sort by difference descending (biggest differences first)
    differences.sort(key=lambda x: x[3], reverse=True)
    
    return differences


def get_problem_description(data: Dict) -> str:
    """Extract a brief description from the problem data."""
    if not data or 'prompt' not in data:
        return "No description available"
    
    # Get first meaningful line from prompt
    prompt_lines = data['prompt'].strip().split('\n')
    description = next((line.strip() for line in prompt_lines 
                       if line.strip() and not line.strip().startswith('--') 
                       and not line.strip().startswith('#') 
                       and not line.strip().startswith('"""')
                       and not line.strip().startswith("'''")
                       and len(line.strip()) > 10), "No description")
    
    if len(description) > 100:
        description = description[:97] + "..."
    
    return description


def get_language_name(lang_code: str) -> str:
    """Get full language name from code."""
    language_names = {
        'jl': 'Julia',
        'lua': 'Lua', 
        'ml': 'OCaml',
        'r': 'R',
        'rkt': 'Racket'
    }
    return language_names.get(lang_code.lower(), lang_code.upper())


def save_results(differences: List, output_dir: str, lang1: str, lang2: str):
    """Save the analysis results to a file."""
    os.makedirs(output_dir, exist_ok=True)
    
    lang1_name = get_language_name(lang1)
    lang2_name = get_language_name(lang2)
    
    output_file = os.path.join(output_dir, f"{lang1}_vs_{lang2}_accuracy_differences.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{lang1_name.upper()} vs {lang2_name.upper()} ACCURACY DIFFERENCES\n")
        f.write("=" * 60 + "\n")
        f.write(f"Problems where {lang1_name} performed better than {lang2_name}, sorted by accuracy difference\n\n")
        
        f.write("Summary:\n")
        f.write(f"Total problems where {lang1_name} > {lang2_name}: {len(differences)}\n")
        if differences:
            f.write(f"Largest difference: {differences[0][3]:.3f}\n")
            f.write(f"Smallest difference: {differences[-1][3]:.3f}\n")
            avg_diff = sum(d[3] for d in differences) / len(differences)
            f.write(f"Average difference: {avg_diff:.3f}\n")
        f.write("\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 20 + "\n\n")
        
        for i, (filename, lang1_acc, lang2_acc, diff, lang1_data, lang2_data) in enumerate(differences, 1):
            f.write(f"{i:2d}. {filename}\n")
            f.write(f"    Accuracy: {lang1_name} {lang1_acc:.3f} | {lang2_name} {lang2_acc:.3f} | Difference: +{diff:.3f}\n")
            
            problem_name = lang1_data.get('name', 'Unknown')
            f.write(f"    Problem: {problem_name}\n")
            
            description = get_problem_description(lang1_data)
            f.write(f"    Description: {description}\n")
            
            # Show test results summary
            lang1_results = lang1_data.get('results', [])
            lang2_results = lang2_data.get('results', [])
            
            if lang1_results:
                lang1_success = sum(1 for r in lang1_results if r.get('status') == 'OK' and r.get('exit_code') == 0)
                f.write(f"    {lang1_name} results: {lang1_success}/{len(lang1_results)} passed\n")
            
            if lang2_results:
                lang2_success = sum(1 for r in lang2_results if r.get('status') == 'OK' and r.get('exit_code') == 0)
                f.write(f"    {lang2_name} results: {lang2_success}/{len(lang2_results)} passed\n")
            
            f.write("\n")
    
    print(f"Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Find problems where one language performed better than another',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_language_accuracy_differences.py /path/to/results lua ml
  python find_language_accuracy_differences.py /path/to/results jl r --output ./analysis
        """
    )
    parser.add_argument('base_dir', help='Base directory containing language subdirectories')
    parser.add_argument('lang1', help='First language (e.g., lua, ml, jl, r, rkt)')
    parser.add_argument('lang2', help='Second language (e.g., lua, ml, jl, r, rkt)')
    parser.add_argument('--output', '-o', default='/home/junsoo/MultiPL-E/analyze_failure', 
                       help='Output directory for results (default: /home/junsoo/MultiPL-E/analyze_failure)')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    lang1 = args.lang1.lower()
    lang2 = args.lang2.lower()
    output_dir = args.output
    
    lang1_name = get_language_name(lang1)
    lang2_name = get_language_name(lang2)
    
    print(f"Comparing {lang1_name} vs {lang2_name}")
    print("=" * 50)
    
    # Analyze first language directory
    lang1_dir = os.path.join(base_dir, lang1)
    print(f"Analyzing {lang1_name} results in: {lang1_dir}")
    lang1_results = analyze_directory(lang1_dir, lang1_name)
    
    # Analyze second language directory  
    lang2_dir = os.path.join(base_dir, lang2)
    print(f"Analyzing {lang2_name} results in: {lang2_dir}")
    lang2_results = analyze_directory(lang2_dir, lang2_name)
    
    if not lang1_results:
        print(f"No {lang1_name} results found!")
        return
    
    if not lang2_results:
        print(f"No {lang2_name} results found!")
        return
    
    # Find differences
    print(f"\nFinding problems where {lang1_name} > {lang2_name}...")
    differences = find_differences(lang1_results, lang2_results, lang1, lang2)
    
    print(f"Found {len(differences)} problems where {lang1_name} performed better than {lang2_name}")
    
    if differences:
        print(f"Largest difference: {differences[0][3]:.3f} ({differences[0][0]})")
        print(f"Top 5 problems:")
        for i, (filename, lang1_acc, lang2_acc, diff, _, _) in enumerate(differences[:5], 1):
            print(f"  {i}. {filename}: {lang1_name} {lang1_acc:.3f} vs {lang2_name} {lang2_acc:.3f} (diff: +{diff:.3f})")
    
    # Save results
    output_file = save_results(differences, output_dir, lang1, lang2)
    
    print(f"\nAnalysis complete! Check {output_file} for detailed results.")


if __name__ == "__main__":
    main()