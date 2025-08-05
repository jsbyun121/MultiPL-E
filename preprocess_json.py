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

def prefix_comments():
    """Define comment start patterns for different languages."""
    return {
        'ml': ['(*', '*'],
        'jl': ['"""'],
        'lua': ['--'],
        'r': ['#'],
        'rkt': [';;'],
    }

def postfix_comments():
    """Define comment end patterns for different languages."""
    return {
        'ml': ['*)'],
        'jl': ['"""'],
        'lua': [],
        'r': [],
        'rkt': [],
    }

def _clean_code(completion):
        """Clean up chat template completions to extract just the code"""
        import re

        # Extract code from markdown blocks if present
        code_block_match = re.search(r'```(?:\w+)?\s*\n(.*?)\n?```', completion, re.DOTALL)
        if code_block_match:
            extracted_code = code_block_match.group(1).strip()

            # breakpoint()
            
            # Remove duplicate #lang racket if it already exists in prompt
            lines = extracted_code.split('\n')

            while len(lines) > 0 and lines[0].strip() == '':
                lines = lines[1:]

            # breakpoint()

            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.lower().startswith(('#lang', '#julia', '#r', '#rkt', '#ocaml', '#lua')):
                    lines.pop(i)
                    break
            
            result = '\n'.join(lines)

            # breakpoint()
            
            return result
        
        # Fallback: clean up the completion as-is
        result = ""
        return result
        
def process_raw_completions(raw_completions: List[str]) -> List[str]:
    cleaned_completions = []
    for raw_completion in raw_completions:
        # Check if </think> exists before splitting
        if "</think>" in raw_completion:
            raw_completion = raw_completion.split("</think>", 1)[1]

        raw_completion = _clean_code(raw_completion)
        cleaned_completions.append(raw_completion)

    return cleaned_completions

def remove_code_from_bottom(prompt: str, language: str) -> str:
    """
    Search from bottom up. Keep everything until we find a comment line.
    """
    lines = prompt.split('\n')
    if not lines:
        return ""

    start_patterns = prefix_comments().get(language, [])
    end_patterns = postfix_comments().get(language, [])


    # Search from bottom up
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Check if this line is a comment
        for start_pattern in start_patterns:
            if stripped.startswith(start_pattern):
                # Found comment - keep everything up to and including this line
                return '\n'.join(lines[:i + 1])
            
            else:
                for end_pattern in end_patterns:
                    if stripped.endswith(end_pattern):
                        # Found comment - keep everything up to and including this line
                        return '\n'.join(lines[:i + 1])

    # No comments found, return original
    return '\n'.join(lines)

def process_json_file(input_path: Path, output_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """Process a single JSON file to separate signature from prompt"""
    stats = {"processed": 0, "signature removed": 0, "errors": 0}
    
    try:
        # Read the file (handle both .json and .json.gz)
        if input_path.suffix == '.gz':
            with gzip.open(input_path, 'rt') as f:
                data = json.load(f)
        else:
            with open(input_path, 'r') as f:
                data = json.load(f)
        
        stats["processed"] = 1
        
        # Remove code from bottom, keeping headers and comments
        original_prompt = data.get("prompt", "")
        language = data.get("language", "")
        raw_completions = data.get("raw_completions", "")
        
        processed_prompt = remove_code_from_bottom(original_prompt, language) + '\n'
        processed_completions = process_raw_completions(raw_completions) if raw_completions else ""
        
        if processed_prompt != original_prompt:
            # Modify the data structure
            data["prompt"] = processed_prompt
            data["completions"] = processed_completions
            stats["modified"] = 1
            
            print(f"Modified {input_path.name}:")
            print(f"  Language: {language}")
            print(f"  Original prompt lines: {len(original_prompt.splitlines())}")
            print(f"  Processed prompt lines: {len(processed_prompt.splitlines())}")
            print(f"  Removed code from bottom, kept headers and comments")
            print()
        
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
        stats["errors"] = 1
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Preprocess MultiPL-E JSON files to separate signature from prompt")
    parser.add_argument("input_dir", type=Path, help="Input directory containing JSON files")
    parser.add_argument("output_dir", type=Path, help="Output directory for processed files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--pattern", type=str, default="**/*.json*", help="File pattern to match")
    parser.add_argument("--language", type=str, help="Process only files with specific language")
    
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
    
    total_stats = {"processed": 0, "modified": 0, "errors": 0}
    
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
        stats = process_json_file(json_file, output_path, args.dry_run)
        
        for key in total_stats:
            total_stats[key] += stats[key]
    
    print(f"\nSummary:")
    print(f"  Files processed (prompts): {total_stats['processed']}")
    print(f"  Files modified (prompts): {total_stats['modified']}")
    print(f"  Errors: {total_stats['errors']}")
    
    if args.dry_run:
        print(f"\nTo apply changes, run without --dry-run flag")
    
    return 0

if __name__ == "__main__":
    exit(main())