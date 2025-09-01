#!/usr/bin/env python3
"""
Script to preprocess MultiPL-E JSON files to separate function signature from prompt.
This eliminates the signature duplication problem at the source.
"""

import json
import gzip
import re
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional
from typing import List
from pprint import pprint
import re

def _clean_code(completion, function_signature):
        """Clean up chat template completions to extract just the code"""

        completion_split = completion.split('\n')

        while completion_split:
            line = completion_split[0]
            # Check if the line contains the function signature
            if function_signature in line:
                # If found, we can use this line
                _, _, after = line.partition(function_signature)
                completion_split[0] = after
                break
            else:
                completion_split = completion_split[1:]

        results = []

        for line in completion_split:
            markdown = "```"
            if markdown in line:
                before, _, _ = line.partition(markdown)
                results.append(before)
                break
            else:
                results.append(line)

        result = '\n'.join(results)
        
        return result

def process_post_completions(post_completions: List[str], function_signature: str, model: str) -> List[str]:
    cleaned_completions = []
    for post_completion in post_completions:
        post_completion = remove_until_end_reasoning(post_completion, model)
        post_completion = _clean_code(post_completion, function_signature)
        cleaned_completions.append(post_completion)

    return cleaned_completions

def remove_until_end_reasoning(post_completion: str, model: str) -> str:
    """Remove text before the end of reasoning marker based on model type."""
    model = model.lower()

    if "qwen" in model and "think" in model:
        # Qwen model: return texts after </think>
        end_reasoning_pattern = r"</think>"
    elif "gpt" in model and "oss" in model:
        # OpenAI model: <|end|><|start|>assistant<|channel|>final<|message|>
        end_reasoning_pattern = r"<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>"
    elif model == "":
        raise ValueError("You must put a model name as arg")
    else:
        return post_completion
    
    match = re.search(end_reasoning_pattern, post_completion)
    if match:
        return post_completion[match.start():]
    return post_completion

def get_function_signature(prompt: str) -> str:
    """Extract the function signature from the prompt."""
    prompt_str = prompt.strip()

    if not prompt_str:
        return ""
    else:
        prompt_split = prompt_str.split('\n')
        function_signature = prompt_split[-1]
        return function_signature.strip()

def process_json_file(input_path: Path, output_path: Path, dry_run: bool = False, model: str = ""):
    """Process a single JSON file to separate signature from prompt"""
    
    try:
        # Read the file (handle both .json and .json.gz)
        if input_path.suffix == '.gz':
            with gzip.open(input_path, 'rt') as f:
                data = json.load(f)
        else:
            with open(input_path, 'r') as f:
                data = json.load(f)
        
        # Remove code from bottom, keeping headers and comments
        original_prompt = data.get("prompt", "")
        language = data.get("language", "")
        post_completions = data.get("post_completions", "")

        function_signature = get_function_signature(original_prompt)
        processed_completions = process_post_completions(post_completions, function_signature, model) if post_completions else ""

        data["completions"] = processed_completions

        for key in ["post_completions", "pre_completions"]:
            if key in data:
                del data[key]
        
        # Write the modified file if not dry run
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.gz':
                with gzip.open(output_path, 'wt') as f:
                    json.dump(data, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess MultiPL-E JSON files to separate signature from prompt")
    parser.add_argument("input_dir", type=Path, help="Input directory containing JSON files")
    parser.add_argument("output_dir", type=Path, help="Output directory for processed files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--pattern", type=str, default="**/*.json*", help="File pattern to match")
    parser.add_argument("--language", type=str, help="Process only files with specific language")
    parser.add_argument("--model", "-m", type=str, help="Give information about the model")

    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    # Find all JSON files
    json_files = list(args.input_dir.glob(args.pattern))
    json_files = [f for f in json_files if not f.name.endswith('.results.json') and not f.name.endswith('.results.json.gz')]
    
    if not json_files:
        print(f"No JSON files found in {args.input_dir} with pattern {args.pattern}")
        return 1
    
    print(f"Found {len(json_files)} JSON files to process")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    print()
    
    for json_file in json_files:
        # Filter by language if specified
        if args.language:
            try:
                if json_file.suffix == '.gz':
                    with gzip.open(json_file, 'rt') as f:
                        data = json.load(f)
                else:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                
                if data.get("language") != args.language:
                    continue
            except:
                continue
        
        # Calculate output path
        relative_path = json_file.relative_to(args.input_dir)
        output_path = args.output_dir / relative_path
        
        # Process the file
        process_json_file(json_file, output_path, args.dry_run, args.model)
    
    if args.dry_run:
        print(f"\nTo apply changes, run without --dry-run flag")
    
    return 0

if __name__ == "__main__":
    exit(main())