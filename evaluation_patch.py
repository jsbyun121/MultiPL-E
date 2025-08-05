#!/usr/bin/env python3
"""
Script to patch the evaluation main.py to use the new JSON structure with separated signatures.
"""

import shutil
from pathlib import Path

def patch_evaluation_main():
    """Patch the evaluation main.py to handle the new JSON structure"""
    
    eval_main_path = Path("/data_fast/home/junsoo/MultiPL-E/evaluation/src/main.py")
    backup_path = eval_main_path.with_suffix('.py.backup')
    
    # Create backup
    shutil.copy2(eval_main_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read the original file
    with open(eval_main_path, 'r') as f:
        content = f.read()
    
    # Find and replace the program construction line
    old_line = 'program = problem["prompt"] + problem["completions"][index] + \'\\n\' + problem["tests"]'
    new_line = '''program = problem["prompt"] + problem.get("signature", "") + problem["completions"][index] + '\\n' + problem["tests"]'''
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print("Patched program construction line")
    else:
        print("Warning: Could not find the exact line to patch. Manual modification may be needed.")
        print("Look for line 34 in main.py and modify it to use signature field")
    
    # Write the patched file
    with open(eval_main_path, 'w') as f:
        f.write(content)
    
    print(f"Patched {eval_main_path}")
    print("The evaluation system now supports the new JSON structure with separated signatures.")

if __name__ == "__main__":
    patch_evaluation_main()