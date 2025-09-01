import re

str = "end"

if __name__ == "__main__":
    end_reasoning_pattern = r"<|end|><|start|>assistant<|channel|>final<|message|>"
    match = re.search(end_reasoning_pattern, str)
    if match:
        print(str[match.start():])
    else:
        print("No match found")