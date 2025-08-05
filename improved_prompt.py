system_msg = """You are a code completion assistant. Analyze the function signature, docstring, and examples carefully. Generate the missing implementation.

Requirements:
- Study all examples to understand the expected behavior
- Think step-by-step about the algorithm
- ALWAYS wrap code in markdown blocks with language identifier
- Provide only the implementation body

Format: ```language\n[code]\n```"""

messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": f"Complete this code by analyzing the examples:\n\n{prompt}\n\nWrap in markdown blocks."}
]