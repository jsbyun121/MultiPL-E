#!/usr/bin/env python3
"""
Script to create a detailed summary of the questions the LLM struggles with most.
This will extract the problem descriptions, common failure patterns, and generated code examples.
Enhanced to support all languages and configurations (direct vs rag).
"""

import json
import gzip
import os
import sys
from pathlib import Path
from collections import defaultdict


def load_json_gz(file_path: str) -> dict:
    """Load and parse a gzipped JSON file."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def analyze_failure_patterns(data: dict, language: str) -> dict:
    """Analyze the failure patterns in the test results."""
    if not data or 'results' not in data:
        return {}
    
    failure_analysis = {
        'total_attempts': len(data['results']),
        'successful_attempts': 0,
        'failure_patterns': {},
        'common_errors': []
    }
    
    for result in data['results']:
        if result.get('status') == 'OK' and result.get('exit_code') == 0:
            failure_analysis['successful_attempts'] += 1
        else:
            # Analyze failure pattern
            status = result.get('status', 'Unknown')
            exit_code = result.get('exit_code', -1)
            stderr = result.get('stderr', '')
            stdout = result.get('stdout', '')
            
            # Count failure types
            pattern_key = f"{status}_exit_{exit_code}"
            if pattern_key not in failure_analysis['failure_patterns']:
                failure_analysis['failure_patterns'][pattern_key] = 0
            failure_analysis['failure_patterns'][pattern_key] += 1
            
            # Extract common error messages (language-specific)
            combined_output = (stdout + stderr).lower()
            if 'failed tests' in stdout.lower() or 'assertion' in combined_output:
                failure_analysis['common_errors'].append("Test assertion failure")
            elif 'syntaxerror' in combined_output or 'syntax error' in combined_output:
                failure_analysis['common_errors'].append("Syntax error")
            elif 'nameerror' in combined_output or 'undefvarerror' in combined_output:
                failure_analysis['common_errors'].append("Undefined variable error")
            elif 'typeerror' in combined_output or 'methoderror' in combined_output:
                failure_analysis['common_errors'].append("Type/method error")
            elif 'attempt to call a nil value' in stdout:
                failure_analysis['common_errors'].append("Nil value call error")
            elif 'attempt to index' in stdout:
                failure_analysis['common_errors'].append("Index error")
            elif 'compilation error' in combined_output:
                failure_analysis['common_errors'].append("Compilation error")
            elif combined_output.strip() == '':
                failure_analysis['common_errors'].append("No output generated")
            else:
                failure_analysis['common_errors'].append("Other error")
    
    return failure_analysis


def extract_function_code(program_text: str, language: str) -> str:
    """Extract the main function from the generated program based on language."""
    lines = program_text.split('\n')
    function_lines = []
    
    if language == 'jl':  # Julia
        in_function = False
        for line in lines:
            if 'function ' in line and not in_function:
                in_function = True
                function_lines.append(line)
            elif in_function and line.strip() == 'end':
                function_lines.append(line)
                break
            elif in_function:
                function_lines.append(line)
    
    elif language == 'lua':  # Lua
        in_function = False
        for line in lines:
            if 'local function' in line and not in_function:
                in_function = True
                function_lines.append(line)
            elif in_function and line.strip() == 'end':
                function_lines.append(line)
                break
            elif in_function:
                function_lines.append(line)
    
    elif language == 'ml':  # OCaml
        in_function = False
        for line in lines:
            if 'let ' in line and '=' in line and not in_function:
                in_function = True
                function_lines.append(line)
            elif in_function and (line.strip() == '' or (line and not line[0].isspace())):
                if line.strip() and not line.startswith('let '):
                    break
            elif in_function:
                function_lines.append(line)
    
    elif language == 'r':  # R
        in_function = False
        for line in lines:
            if '<- function(' in line and not in_function:
                in_function = True
                function_lines.append(line)
            elif in_function and line.strip() == '}':
                function_lines.append(line)
                break
            elif in_function:
                function_lines.append(line)
    
    elif language == 'rkt':  # Racket
        in_function = False
        paren_count = 0
        for line in lines:
            if '(define ' in line and not in_function:
                in_function = True
                function_lines.append(line)
                paren_count = line.count('(') - line.count(')')
            elif in_function:
                function_lines.append(line)
                paren_count += line.count('(') - line.count(')')
                if paren_count <= 0:
                    break
    
    if function_lines:
        return '\n'.join(function_lines)
    
    # Fallback: return first 20 lines if function extraction fails
    return '\n'.join(lines[:20])


def get_test_cases(data: dict, language: str) -> list:
    """Extract test cases based on language."""
    if 'tests' not in data:
        return []
    
    test_lines = data['tests'].strip().split('\n')
    test_cases = []
    
    if language == 'lua':
        for line in test_lines:
            if 'lu.assertEquals' in line:
                test_cases.append(line.strip())
    elif language == 'jl':
        for line in test_lines:
            if 'Test.@test' in line or '@test' in line:
                test_cases.append(line.strip())
    elif language == 'ml':
        for line in test_lines:
            if 'assert' in line or '=' in line and 'let' in line:
                test_cases.append(line.strip())
    elif language == 'r':
        for line in test_lines:
            if 'stopifnot' in line or 'testthat::expect' in line:
                test_cases.append(line.strip())
    elif language == 'rkt':
        for line in test_lines:
            if 'check-equal?' in line or 'test' in line:
                test_cases.append(line.strip())
    
    return test_cases[:5]  # Limit to first 5 test cases


def find_universal_zero_accuracy_files(base_result_path: str):
    """Find files with 0% accuracy across all languages."""
    languages = ['jl', 'lua', 'ml', 'r', 'rkt']
    zero_accuracy_files = defaultdict(set)
    
    for lang in languages:
        lang_path = os.path.join(base_result_path, lang)
        if not os.path.exists(lang_path):
            continue
            
        json_files = Path(lang_path).glob("*.json.gz")
        
        for json_file in json_files:
            data = load_json_gz(str(json_file))
            if data:
                accuracy = calculate_accuracy(data)
                if accuracy == 0.0:
                    zero_accuracy_files[lang].add(json_file.name)
    
    # Find intersection across all languages
    if zero_accuracy_files:
        universal_zero_files = set(zero_accuracy_files[languages[0]])
        for lang in languages[1:]:
            universal_zero_files = universal_zero_files.intersection(zero_accuracy_files[lang])
    else:
        universal_zero_files = set()
    
    return sorted(universal_zero_files)


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


def create_detailed_summary(language: str, config_type: str, base_result_path: str):
    """Create a detailed summary of universal zero accuracy questions."""
    
    language_names = {
        'jl': 'Julia',
        'lua': 'Lua',
        'ml': 'OCaml', 
        'r': 'R',
        'rkt': 'Racket'
    }
    
    lang_name = language_names.get(language, language.upper())
    
    # Find universal zero accuracy files
    universal_zero_files = find_universal_zero_accuracy_files(base_result_path)
    
    if not universal_zero_files:
        print(f"No universal zero accuracy files found for {lang_name}!")
        return ""
    
    base_path = Path(base_result_path) / language
    
    summary_content = []
    summary_content.append("=" * 80)
    summary_content.append(f"DETAILED ANALYSIS: UNIVERSAL ZERO ACCURACY CASES - {lang_name.upper()}")
    summary_content.append("=" * 80)
    summary_content.append("")
    summary_content.append(f"These {len(universal_zero_files)} questions achieved 0% accuracy across ALL 5 languages.")
    summary_content.append(f"Analysis focuses on {lang_name} implementations with generated code examples.")
    summary_content.append(f"Configuration: {config_type.upper()}")
    summary_content.append("")
    
    for i, filename in enumerate(universal_zero_files, 1):
        file_path = base_path / filename
        data = load_json_gz(str(file_path))
        
        if not data:
            continue
            
        summary_content.append(f"{'='*60}")
        summary_content.append(f"QUESTION #{i}: {data.get('name', 'Unknown')}")
        summary_content.append(f"{'='*60}")
        summary_content.append("")
        
        # Problem description
        summary_content.append("PROBLEM DESCRIPTION:")
        summary_content.append("-" * 20)
        if 'prompt' in data:
            prompt_lines = data['prompt'].strip().split('\n')
            for line in prompt_lines[:8]:  # First 8 lines
                if line.strip():
                    summary_content.append(f"  {line}")
        summary_content.append("")
        
        # Test cases (language-aware)
        test_cases = get_test_cases(data, language)
        if test_cases:
            summary_content.append("TEST CASES:")
            summary_content.append("-" * 11)
            for test_case in test_cases:
                summary_content.append(f"  {test_case}")
            summary_content.append("")
        
        # Failure analysis
        failure_analysis = analyze_failure_patterns(data, language)
        summary_content.append("FAILURE ANALYSIS:")
        summary_content.append("-" * 16)
        summary_content.append(f"  Total attempts: {failure_analysis['total_attempts']}")
        summary_content.append(f"  Successful: {failure_analysis['successful_attempts']}")
        summary_content.append(f"  Failed: {failure_analysis['total_attempts'] - failure_analysis['successful_attempts']}")
        summary_content.append("")
        
        if failure_analysis['failure_patterns']:
            summary_content.append("  Failure patterns:")
            for pattern, count in failure_analysis['failure_patterns'].items():
                summary_content.append(f"    - {pattern}: {count} times")
        
        if failure_analysis['common_errors']:
            summary_content.append("  Common error types:")
            error_counts = {}
            for error in failure_analysis['common_errors']:
                error_counts[error] = error_counts.get(error, 0) + 1
            for error, count in error_counts.items():
                summary_content.append(f"    - {error}: {count} times")
        summary_content.append("")
        
        # Show up to 2 example failures with generated code
        if 'results' in data and data['results']:
            examples_shown = 0
            for j, result in enumerate(data['results']):
                if result.get('status') != 'OK' or result.get('exit_code') != 0:
                    if examples_shown >= 2:  # Limit to 2 examples
                        break
                    
                    examples_shown += 1
                    summary_content.append(f"EXAMPLE FAILURE #{examples_shown}:")
                    summary_content.append("-" * 20)
                    
                    # Extract and show generated function
                    program = result.get('program', '')
                    function_code = extract_function_code(program, language)
                    summary_content.append("Generated Function:")
                    for line in function_code.split('\n')[:12]:  # First 12 lines
                        summary_content.append(f"  {line}")
                    
                    if len(function_code.split('\n')) > 12:
                        summary_content.append("  ... (truncated)")
                    
                    # Error details
                    summary_content.append("")
                    summary_content.append(f"Status: {result.get('status', 'Unknown')}")
                    summary_content.append(f"Exit Code: {result.get('exit_code', 'Unknown')}")
                    
                    # Show error output
                    stdout = result.get('stdout', '')
                    if stdout and len(stdout.strip()) > 0:
                        summary_content.append("Error Output:")
                        error_lines = stdout.strip().split('\n')[:4]  # First 4 lines
                        for line in error_lines:
                            if line.strip():
                                summary_content.append(f"  {line.strip()}")
                    
                    summary_content.append("")
        
        summary_content.append("")
    
    # Overall analysis
    summary_content.append("=" * 80)
    summary_content.append(f"OVERALL PATTERNS AND INSIGHTS - {lang_name.upper()}")
    summary_content.append("=" * 80)
    summary_content.append("")
    summary_content.append("Common issues observed across these universal failing questions:")
    summary_content.append("")
    
    # Language-specific insights
    if language == 'lua':
        summary_content.append("1. Lua-specific syntax issues:")
        summary_content.append("   - Use of deprecated functions like table.getn() instead of # operator")
        summary_content.append("   - Incorrect table manipulation methods")
        summary_content.append("   - Missing return statements or incorrect return values")
    elif language == 'jl':
        summary_content.append("1. Julia-specific syntax issues:")
        summary_content.append("   - Array indexing errors (1-based vs 0-based confusion)")
        summary_content.append("   - Type annotation problems")
        summary_content.append("   - Broadcasting and vectorization issues")
    elif language == 'ml':
        summary_content.append("1. OCaml-specific syntax issues:")
        summary_content.append("   - Pattern matching syntax errors")
        summary_content.append("   - Type inference problems")
        summary_content.append("   - List manipulation and recursion issues")
    elif language == 'r':
        summary_content.append("1. R-specific syntax issues:")
        summary_content.append("   - Vector indexing problems (1-based indexing)")
        summary_content.append("   - Data frame manipulation errors")
        summary_content.append("   - Function definition and scoping issues")
    elif language == 'rkt':
        summary_content.append("1. Racket-specific syntax issues:")
        summary_content.append("   - S-expression and parentheses balancing problems")
        summary_content.append("   - List processing and recursion errors")
        summary_content.append("   - Function definition syntax issues")
    
    summary_content.append("")
    summary_content.append("2. Logic errors:")
    summary_content.append("   - Incorrect algorithm implementation")
    summary_content.append("   - Off-by-one errors in loops")
    summary_content.append("   - Misunderstanding of problem requirements")
    summary_content.append("")
    summary_content.append("3. Edge case handling:")
    summary_content.append("   - Poor handling of empty collections")
    summary_content.append("   - Incorrect handling of special input values")
    summary_content.append("   - Missing validation for boundary conditions")
    summary_content.append("")
    summary_content.append(f"These patterns suggest the LLM needs better training on:")
    summary_content.append(f"- Modern {lang_name} syntax and idioms")
    summary_content.append(f"- Algorithm implementation in {lang_name}")
    summary_content.append(f"- Edge case handling and input validation")
    summary_content.append(f"- Language-specific data structure manipulation")
    
    return '\n'.join(summary_content)


def main():
    """Main function that can be called with command line arguments or defaults."""
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        language = sys.argv[1]
        config_type = sys.argv[2]
    else:
        # Default values
        language = 'lua'
        config_type = 'rag'
    
    # Validate inputs
    valid_languages = ['jl', 'lua', 'ml', 'r', 'rkt']
    valid_configs = ['direct', 'rag']
    
    if language not in valid_languages:
        print(f"Error: Language must be one of {valid_languages}")
        return
    
    if config_type not in valid_configs:
        print(f"Error: Config type must be one of {valid_configs}")
        return
    
    # Determine the result path based on config type
    if config_type == 'rag':
        base_result_path = "/home/junsoo/MultiPL-E/after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024_rag_3/result"
    else:
        base_result_path = "/home/junsoo/MultiPL-E/after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024/result"
    
    language_names = {
        'jl': 'Julia',
        'lua': 'Lua',
        'ml': 'OCaml', 
        'r': 'R',
        'rkt': 'Racket'
    }
    
    lang_name = language_names[language]
    
    print(f"Creating detailed summary for {lang_name} ({language}) - {config_type.upper()} configuration...")
    
    # Create summary
    summary = create_detailed_summary(language, config_type, base_result_path)
    
    if not summary:
        print("No summary generated!")
        return
    
    # Save detailed analysis
    output_dir = f"/home/junsoo/MultiPL-E/analyze_failure/{config_type}/{language}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "struggle_questions_detailed_analysis.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Detailed analysis saved to: {output_file}")
    
    # Create the simple filename list 
    universal_zero_files = find_universal_zero_accuracy_files(base_result_path)
    simple_list_file = os.path.join(output_dir, "14_universal_zero_accuracy_filenames.txt")
    with open(simple_list_file, 'w', encoding='utf-8') as f:
        f.write(f"Universal Zero Accuracy Cases - {lang_name} ({config_type.upper()})\n")
        f.write("=" * 60 + "\n\n")
        f.write("These questions achieved 0% accuracy across ALL 5 languages.\n\n")
        
        for i, filename in enumerate(universal_zero_files, 1):
            f.write(f"{i:2d}. {filename}\n")
        
        f.write(f"\nTotal: {len(universal_zero_files)} universal failure cases\n")
    
    print(f"Filename list saved to: {simple_list_file}")
    print(f"\nUsage: python3 {sys.argv[0]} <language> <config_type>")
    print(f"Languages: {', '.join(valid_languages)}")
    print(f"Configs: {', '.join(valid_configs)}")


if __name__ == "__main__":
    main()