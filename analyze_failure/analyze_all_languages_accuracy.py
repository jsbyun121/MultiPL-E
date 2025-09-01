#!/usr/bin/env python3
"""
Comprehensive script to analyze JSON.gz files across all 5 languages and find 0% accuracy cases.
Languages: Julia (jl), Lua (lua), OCaml (ml), R (r), Racket (rkt)
"""

import json
import gzip
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


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
        # Check if the test passed (status == "OK" and exit_code == 0)
        if result.get('status') == 'OK' and result.get('exit_code') == 0:
            successful_count += 1
    
    accuracy = successful_count / total_count if total_count > 0 else 0.0
    return accuracy


def analyze_language_directory(language: str, directory_path: str) -> List[Tuple[str, float, Dict]]:
    """Analyze all JSON.gz files in a language directory and return accuracy data."""
    results = []
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return results
    
    # Find all .json.gz files
    json_files = list(directory.glob("*.json.gz"))
    print(f"  {language.upper()}: Found {len(json_files)} JSON.gz files")
    
    for json_file in json_files:
        data = load_json_gz(str(json_file))
        
        if data is not None:
            accuracy = calculate_accuracy(data)
            results.append((json_file.name, accuracy, data))
    
    return results


def get_zero_accuracy_cases(results: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float, Dict]]:
    """Get all cases with 0% accuracy."""
    return [(filename, accuracy, data) for filename, accuracy, data in results if accuracy == 0.0]


def analyze_failure_patterns(data: Dict) -> Dict:
    """Analyze the failure patterns in the test results."""
    if not data or 'results' not in data:
        return {}
    
    failure_analysis = {
        'total_attempts': len(data['results']),
        'successful_attempts': 0,
        'failure_types': defaultdict(int),
        'common_errors': []
    }
    
    for result in data['results']:
        if result.get('status') == 'OK' and result.get('exit_code') == 0:
            failure_analysis['successful_attempts'] += 1
        else:
            # Analyze failure pattern
            status = result.get('status', 'Unknown')
            exit_code = result.get('exit_code', -1)
            stdout = result.get('stdout', '')
            
            # Count failure types
            failure_type = f"{status}_exit_{exit_code}"
            failure_analysis['failure_types'][failure_type] += 1
            
            # Extract common error messages
            if 'Failed tests' in stdout:
                failure_analysis['common_errors'].append("Test assertion failure")
            elif 'attempt to call a nil value' in stdout:
                failure_analysis['common_errors'].append("Nil value call error")
            elif 'syntax error' in stdout.lower() or 'SyntaxError' in stdout:
                failure_analysis['common_errors'].append("Syntax error")
            elif 'attempt to index' in stdout:
                failure_analysis['common_errors'].append("Index error")
            elif 'NameError' in stdout or 'UndefVarError' in stdout:
                failure_analysis['common_errors'].append("Undefined variable error")
            elif 'TypeError' in stdout or 'MethodError' in stdout:
                failure_analysis['common_errors'].append("Type/method error")
            elif 'compilation error' in stdout.lower():
                failure_analysis['common_errors'].append("Compilation error")
            elif stdout.strip() == '':
                failure_analysis['common_errors'].append("No output generated")
    
    return failure_analysis


def main():
    # Base directory containing language subdirectories
    base_path = "/home/junsoo/MultiPL-E/after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024_rag_3/result"
    
    # Language mappings
    languages = {
        'jl': 'Julia',
        'lua': 'Lua', 
        'ml': 'OCaml',
        'r': 'R',
        'rkt': 'Racket'
    }
    
    print("Starting comprehensive analysis across all languages...")
    print("=" * 70)
    
    all_results = {}
    zero_accuracy_summary = {}
    language_stats = {}
    
    # Analyze each language
    for lang_code, lang_name in languages.items():
        print(f"\nAnalyzing {lang_name} ({lang_code})...")
        directory_path = os.path.join(base_path, lang_code)
        
        results = analyze_language_directory(lang_code, directory_path)
        all_results[lang_code] = results
        
        if results:
            # Calculate language statistics
            accuracies = [acc for _, acc, _ in results]
            zero_accuracy_cases = get_zero_accuracy_cases(results)
            
            language_stats[lang_code] = {
                'name': lang_name,
                'total_files': len(results),
                'zero_accuracy_count': len(zero_accuracy_cases),
                'average_accuracy': sum(accuracies) / len(accuracies),
                'min_accuracy': min(accuracies),
                'max_accuracy': max(accuracies)
            }
            
            zero_accuracy_summary[lang_code] = zero_accuracy_cases
            
            print(f"  Total files: {len(results)}")
            print(f"  Zero accuracy cases: {len(zero_accuracy_cases)}")
            print(f"  Average accuracy: {language_stats[lang_code]['average_accuracy']:.3f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 30)
    for lang_code, stats in language_stats.items():
        print(f"{stats['name']:<8} | Files: {stats['total_files']:3d} | "
              f"0% accuracy: {stats['zero_accuracy_count']:2d} | "
              f"Avg accuracy: {stats['average_accuracy']:.3f}")
    
    # Save detailed analysis for each language
    save_comprehensive_analysis(zero_accuracy_summary, language_stats, languages)
    
    # Save cross-language comparison
    save_cross_language_comparison(zero_accuracy_summary, language_stats, languages)
    
    print(f"\nDetailed analysis files saved in /data_fast/home/junsoo/MultiPL-E/")


def save_comprehensive_analysis(zero_accuracy_summary, language_stats, languages):
    """Save detailed analysis for each language."""
    
    for lang_code, zero_cases in zero_accuracy_summary.items():
        if not zero_cases:
            continue
            
        lang_name = languages[lang_code]
        filename = f"/data_fast/home/junsoo/MultiPL-E/zero_accuracy_{lang_code}_{lang_name.lower()}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"ZERO ACCURACY CASES FOR {lang_name.upper()} ({lang_code})\n")
            f.write("=" * 60 + "\n\n")
            
            stats = language_stats[lang_code]
            f.write(f"Language Statistics:\n")
            f.write(f"  Total files analyzed: {stats['total_files']}\n")
            f.write(f"  Zero accuracy cases: {stats['zero_accuracy_count']}\n")
            f.write(f"  Percentage with 0% accuracy: {(stats['zero_accuracy_count']/stats['total_files']*100):.1f}%\n")
            f.write(f"  Average accuracy: {stats['average_accuracy']:.3f}\n\n")
            
            f.write("Zero Accuracy Questions:\n")
            f.write("-" * 25 + "\n")
            for i, (filename, accuracy, data) in enumerate(zero_cases, 1):
                f.write(f"{i:2d}. {filename}\n")
                f.write(f"    Problem: {data.get('name', 'Unknown')}\n")
                if 'prompt' in data:
                    # Get first line of prompt that contains actual description
                    prompt_lines = data['prompt'].strip().split('\n')
                    description = next((line.strip() for line in prompt_lines if line.strip() and not line.strip().startswith('--') and not line.strip().startswith('#')), "No description")
                    if len(description) > 80:
                        description = description[:77] + "..."
                    f.write(f"    Description: {description}\n")
                
                # Analyze failure patterns
                failure_analysis = analyze_failure_patterns(data)
                f.write(f"    Failed attempts: {failure_analysis['total_attempts'] - failure_analysis['successful_attempts']}/{failure_analysis['total_attempts']}\n")
                
                # Show most common error type
                if failure_analysis['common_errors']:
                    from collections import Counter
                    error_counts = Counter(failure_analysis['common_errors'])
                    most_common_error = error_counts.most_common(1)[0]
                    f.write(f"    Main error type: {most_common_error[0]} ({most_common_error[1]} times)\n")
                
                f.write("\n")
        
        print(f"  {lang_name} zero accuracy analysis saved: {os.path.basename(filename)}")


def save_cross_language_comparison(zero_accuracy_summary, language_stats, languages):
    """Save cross-language comparison analysis."""
    
    output_file = "/data_fast/home/junsoo/MultiPL-E/cross_language_zero_accuracy_analysis.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CROSS-LANGUAGE ZERO ACCURACY ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Overview:\n")
        f.write("-" * 10 + "\n")
        f.write("Analysis of questions where the LLM (Qwen3-4B-Instruct) achieved 0% accuracy\n")
        f.write("across 5 programming languages: Julia, Lua, OCaml, R, and Racket.\n\n")
        
        # Summary table
        f.write("Language Comparison Summary:\n")
        f.write("-" * 30 + "\n")
        f.write("Language  | Total | 0% Acc | Percentage | Avg Accuracy\n")
        f.write("-" * 55 + "\n")
        
        for lang_code, stats in language_stats.items():
            pct_zero = (stats['zero_accuracy_count'] / stats['total_files'] * 100)
            f.write(f"{stats['name']:<9} | {stats['total_files']:5d} | {stats['zero_accuracy_count']:6d} | "
                   f"{pct_zero:9.1f}% | {stats['average_accuracy']:11.3f}\n")
        
        f.write("\n")
        
        # Find common problems across languages
        f.write("Analysis by Problem:\n")
        f.write("-" * 20 + "\n")
        
        # Collect all problems that have 0% accuracy in any language
        all_zero_problems = set()
        problem_language_map = defaultdict(list)
        
        for lang_code, zero_cases in zero_accuracy_summary.items():
            for filename, accuracy, data in zero_cases:
                problem_name = data.get('name', 'Unknown')
                all_zero_problems.add(problem_name)
                problem_language_map[problem_name].append(languages[lang_code])
        
        # Sort problems by how many languages they fail in
        sorted_problems = sorted(problem_language_map.items(), 
                               key=lambda x: len(x[1]), reverse=True)
        
        f.write("Problems with 0% accuracy (sorted by number of languages affected):\n\n")
        
        for problem_name, failing_languages in sorted_problems:
            f.write(f"• {problem_name}\n")
            f.write(f"  Fails in {len(failing_languages)} language(s): {', '.join(failing_languages)}\n")
            
            # Get description from any language
            description = "No description available"
            for lang_code, zero_cases in zero_accuracy_summary.items():
                for filename, accuracy, data in zero_cases:
                    if data.get('name') == problem_name and 'prompt' in data:
                        prompt_lines = data['prompt'].strip().split('\n')
                        description = next((line.strip() for line in prompt_lines 
                                          if line.strip() and not line.strip().startswith('--') 
                                          and not line.strip().startswith('#')), "No description")
                        break
                if description != "No description available":
                    break
            
            if len(description) > 100:
                description = description[:97] + "..."
            f.write(f"  Description: {description}\n\n")
        
        # Language-specific insights
        f.write("\nLanguage-Specific Insights:\n")
        f.write("-" * 27 + "\n")
        
        for lang_code, stats in language_stats.items():
            lang_name = stats['name']
            zero_count = stats['zero_accuracy_count']
            total_count = stats['total_files']
            
            f.write(f"\n{lang_name} ({lang_code}):\n")
            if zero_count == 0:
                f.write(f"  Excellent performance - no 0% accuracy cases!\n")
            else:
                pct = (zero_count / total_count * 100)
                f.write(f"  {zero_count} out of {total_count} problems ({pct:.1f}%) with 0% accuracy\n")
                
                # Analyze common error patterns for this language
                if lang_code in zero_accuracy_summary:
                    all_errors = []
                    for _, _, data in zero_accuracy_summary[lang_code]:
                        failure_analysis = analyze_failure_patterns(data)
                        all_errors.extend(failure_analysis['common_errors'])
                    
                    if all_errors:
                        from collections import Counter
                        error_counts = Counter(all_errors)
                        top_errors = error_counts.most_common(3)
                        f.write(f"  Most common error types:\n")
                        for error_type, count in top_errors:
                            f.write(f"    - {error_type}: {count} occurrences\n")
        
        f.write("\nConclusions:\n")
        f.write("-" * 12 + "\n")
        f.write("• Languages with most 0% accuracy cases indicate areas needing LLM improvement\n")
        f.write("• Problems failing across multiple languages suggest fundamental algorithm/logic issues\n")
        f.write("• Language-specific failures indicate syntax or idiom knowledge gaps\n")
        f.write("• This analysis can guide targeted training data improvements for the LLM\n")
    
    print(f"Cross-language comparison saved to: {output_file}")


def save_simple_filenames_all_languages():
    """Create simple filename lists for all languages."""
    base_path = "/home/junsoo/MultiPL-E/after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024_rag_3/result"
    languages = {
        'jl': 'Julia',
        'lua': 'Lua', 
        'ml': 'OCaml',
        'r': 'R',
        'rkt': 'Racket'
    }
    
    output_file = "/data_fast/home/junsoo/MultiPL-E/all_languages_zero_accuracy_filenames.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ZERO ACCURACY FILENAMES ACROSS ALL LANGUAGES\n")
        f.write("=" * 50 + "\n\n")
        
        for lang_code, lang_name in languages.items():
            f.write(f"{lang_name.upper()} ({lang_code}):\n")
            f.write("-" * 20 + "\n")
            
            directory_path = os.path.join(base_path, lang_code)
            results = analyze_language_directory(lang_code, directory_path)
            zero_cases = get_zero_accuracy_cases(results)
            
            if zero_cases:
                for i, (filename, accuracy, data) in enumerate(zero_cases, 1):
                    f.write(f"{i:2d}. {filename}\n")
            else:
                f.write("    No 0% accuracy cases found!\n")
            
            f.write("\n")
    
    print(f"All languages filename list saved to: {output_file}")


if __name__ == "__main__":
    main()
    save_simple_filenames_all_languages()