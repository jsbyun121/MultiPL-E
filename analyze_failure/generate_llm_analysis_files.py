#!/usr/bin/env python3
"""
Generate text files with questions and completions for LLM analysis.
Creates files in format: {lang}-{accuracy}-{question_list}.txt
Example: rkt-25pct-1_3_47_104_107_108_111_126_132_141.txt
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


def format_completion_data(data: dict, problem_num: str, lang: str) -> str:
    """Format completion data for analysis."""
    if not data or 'results' not in data:
        return ""
    
    output = f"\n{'='*80}\n"
    output += f"PROBLEM {problem_num} ({lang.upper()}) - {data.get('name', 'Unknown')}\n"
    output += f"{'='*80}\n"
    
    # Add prompt
    output += f"\nPROMPT:\n{'-'*40}\n"
    output += data.get('prompt', 'No prompt available')
    output += f"\n{'-'*40}\n"
    
    # Add test cases
    output += f"\nTEST CASES:\n{'-'*40}\n"
    output += data.get('tests', 'No tests available')
    output += f"\n{'-'*40}\n"
    
    # Add each completion result
    results = data['results']
    for i, result in enumerate(results, 1):
        output += f"\nCOMPLETION {i}:\n{'-'*20}\n"
        output += f"PROGRAM:\n{result.get('program', 'No program')}\n"
        output += f"\nEXECUTION RESULTS:\n"
        output += f"Exit Code: {result.get('exit_code', 'N/A')}\n"
        output += f"Status: {result.get('status', 'N/A')}\n"
        output += f"Timestamp: {result.get('timestamp', 'N/A')}\n"
        
        stdout = result.get('stdout', '').strip()
        stderr = result.get('stderr', '').strip()
        
        if stdout:
            output += f"STDOUT:\n{stdout}\n"
        else:
            output += "STDOUT: (empty)\n"
            
        if stderr:
            output += f"STDERR:\n{stderr}\n"
        else:
            output += "STDERR: (empty)\n"
        
        output += f"{'-'*20}\n"
    
    return output


def create_analysis_prompt() -> str:
    """Create the analysis prompt for LLM."""
    return """As per my standing instructions, analyze each code-execution result provided. For every item, produce a structured report as follows.

1) Status
Status: "OK" (Success) or else (Failure)
Evidence: include exit code, error status, and whether stderr is empty/non-empty (quote the first relevant line).

If Failure (non-zero exit code, error status, or non-empty stderr)
Root Cause Analysis
Explain why it failed, grounded in the logs/trace.
Point to the exact faulty locations (file and line numbers if available). Quote the minimal offending snippets.
Recommended Fix (Minimal Patch)
Provide brief code segments (3–7 lines each) that would correct the issue.
If multiple faults exist, list fixes in order of impact.
Final Answer
Classify the ultimate source of failure as one of: faulty problem setting / wrong test case / incorrect code implementation.
State it on a single line: Final Answer: <chosen category>.

If Success (exit code 0 and empty stderr)
Critical Correctness Review
Do not assume success means correctness.
Check if the code's logic truly matches the intended problem requirements, including edge cases and corner conditions.
Identify any silent errors, logical flaws, or incompleteness that would still pass execution but fail semantically.
If the implementation is sound, explain why it is correct. If not, explain the discrepancy clearly.
Final Answer
Explicitly confirm or reject correctness in one line: Final Answer: correct or incorrect (with a one-phrase justification if incorrect).

General rules
Be precise and concise. Don't invent behavior not evidenced by the logs or code.
When line numbers aren't available, reference the closest identifiable code block.
Prefer minimal fixes over refactors; include just enough context to apply the patch.

CONTENT TO ANALYZE:
"""


def main():
    base_path = "/home/junsoo/MultiPL-E/after_proc_openai_gpt-oss-20b_mt_4096/result"
    output_dir = "/home/junsoo/MultiPL-E/llm_analysis_files"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Process each language to categorize problems
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
    
    # Define accuracy categories we want to process (3/4 and 4/4)
    accuracy_categories = {
        3: "75pct",  # 75% accuracy
        4: "100pct"  # 100% accuracy
    }
    
    # Generate text files for each language and accuracy category
    for num_trials in [3, 4]:  # Only 3/4 (75%) and 4/4 (100%) categories
        accuracy_label = accuracy_categories[num_trials]
        
        for lang in languages:
            problems = sorted(results_by_trials[num_trials][lang], key=int)
            
            if not problems:
                print(f"No problems found for {lang} with {num_trials}/4 trials correct")
                continue
            
            # Take ALL problems, but chunk them into groups of 10
            all_problems = problems
            
            if not all_problems:
                print(f"No problems available for {lang} with {num_trials}/4 trials correct")
                continue
            
            # Split into chunks of 10 problems
            chunk_size = 10
            for chunk_idx in range(0, len(all_problems), chunk_size):
                chunk_problems = all_problems[chunk_idx:chunk_idx + chunk_size]
                
                # Create filename with chunk info
                chunk_num = (chunk_idx // chunk_size) + 1
                problem_list = "_".join(chunk_problems)
                filename = f"{lang}-{accuracy_label}-chunk{chunk_num}-{problem_list}.txt"
                output_path = os.path.join(output_dir, filename)
                
                print(f"Generating {filename}...")
                
                # Generate content
                content = create_analysis_prompt()
                
                # Add data for each problem in this chunk
                for problem_num in chunk_problems:
                    # Load the specific problem file
                    problem_filename = f"HumanEval_{problem_num}_*.results.json.gz"
                    matching_files = list(Path(os.path.join(base_path, lang)).glob(problem_filename))
                    
                    if matching_files:
                        problem_file = matching_files[0]  # Take the first match
                        problem_data = load_json_gz(str(problem_file))
                        
                        if problem_data:
                            content += format_completion_data(problem_data, problem_num, lang)
                        else:
                            content += f"\n{'='*80}\n"
                            content += f"ERROR: Could not load data for problem {problem_num}\n"
                            content += f"{'='*80}\n"
                    else:
                        content += f"\n{'='*80}\n"
                        content += f"ERROR: Could not find file for problem {problem_num}\n"
                        content += f"{'='*80}\n"
                
                # Write to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                total_completions = len(chunk_problems) * 4
                print(f"  Created {filename} with {len(chunk_problems)} problems ({total_completions} completions)")
            
            total_chunks = (len(all_problems) + chunk_size - 1) // chunk_size
            print(f"  Total for {lang}-{accuracy_label}: {len(all_problems)} problems in {total_chunks} chunks")
    
    print(f"\nAll files generated in: {output_dir}")
    print("\nFile naming format: {lang}-{accuracy}-chunk{N}-{question_numbers}.txt")
    print("Where:")
    print("  - lang: programming language (jl, lua, ml, r, rkt)")
    print("  - accuracy: 75pct or 100pct")
    print("  - chunkN: chunk number (chunk1, chunk2, etc.)")
    print("  - question_numbers: underscore-separated list of HumanEval problem numbers (up to 10 per chunk)")


if __name__ == "__main__":
    main()