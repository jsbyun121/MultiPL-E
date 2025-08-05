"""
This script produces completions for roughly any AutoModelForCausalLM.
"""
from multipl_e.completions import make_main, stop_at_stop_token, partial_arg_parser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from typing import List
from pprint import pprint

class Model:
    def __init__(self, name, revision, model_kwargs, tokenizer_name=None, tokenizer_revision=None,  use_chat_template=True, enable_thinking=True):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype, trust_remote_code=True, **model_kwargs
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or name,
            revision=tokenizer_revision,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        assert (
            self.tokenizer.pad_token is not None
        ), "tokenizer has neither pad_token nor eos_token"

        self._all_special_token_ids = self.tokenizer.all_special_ids
        self.use_chat_template = use_chat_template
        self.enable_thinking = enable_thinking

        assert (
            len(self._all_special_token_ids) >= 1
        ), "tokenizer.all_special_ids() is empty"
        assert (
            self.tokenizer.pad_token_id in self._all_special_token_ids
        ), "pad_token_id not in all_special_ids"
        assert (
            self.tokenizer.eos_token_id in self._all_special_token_ids
        ), "eos_token_id not in all_special_ids"

    def continue_completion_tensor(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.model.eval() # Not essential, but just in case.

        inputs = self.tokenizer(
            [prompt],
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
        ).to("cuda")

        if self.enable_thinking:
            generation_kwargs = {
                **inputs,
                "do_sample": True,
                "use_cache": True,
                "top_p": 0.95,
                "temperature": 0.6,
                "top_k": 20,
                "min_p": 0.0,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id
            }
        else:
            generation_kwargs = {
                **inputs,
                "do_sample": True,
                "use_cache": True,
                "top_p": 0.8,
                "temperature": 0.7,
                "top_k": 20,
                "min_p": 0.0,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id
            }

        with torch.no_grad():
            output = self.model.generate(**generation_kwargs)
        return output

    def completion_tensors(
        self,
        prompts: list,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.model.eval() # Not essential, but just in case.

        formatted_prompts = []

        for prompt in prompts:
            # Optimize system message based on model type
            system_msg = "You are a helpful code completion assistant. Using given examples and the signature, generate the missing implementation by wrapping your code in ```language markdown blocks."

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Complete the code in ```language markdown blocks:\n\n{prompt}\n\n"}
            ]
            
            # Apply chat template with model-specific parameters
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking  # Use instance setting for thinking mode
            )
            formatted_prompts.append(text)
        
        prompts = formatted_prompts
        self._formatted_prompts = formatted_prompts


        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
        ).to("cuda")

        if self.enable_thinking:
            generation_kwargs = {
                **inputs,
                "do_sample": True,
                "use_cache": True,
                "top_p": 0.95,
                "temperature": 0.6,
                "top_k": 20,
                "min_p": 0.0,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id
            }
        else:
            generation_kwargs = {
                **inputs,
                "do_sample": True,
                "use_cache": True,
                "top_p": 0.8,
                "temperature": 0.7,
                "top_k": 20,
                "min_p": 0.0,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id
            }

        with torch.no_grad():
            output = self.model.generate(**generation_kwargs)
        return output


    def _is_special_token_id(self, token_id: int) -> bool:
        """
        Identifies special tokens that should be filtered out, focusing on
        padding and system tokens rather than markdown content tokens.
        """
        # Check for padding and BOS tokens
        if token_id == self.tokenizer.pad_token_id:
            return True
        if self.tokenizer.bos_token_id is not None and token_id == self.tokenizer.bos_token_id:
            return True
            
        # Check for end-of-sequence and other critical special tokens
        if token_id == self.tokenizer.eos_token_id:
            return True
            
        return False

    def _remove_until_endthink_token(self, full_token_ids: List[int]) -> List[int]:
        """
        Removes tokens until the </think> token, which is used to indicate the start of the generated
        response. This is useful for models that use a chat template with thinking mode enabled.
        """
        think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        
        if think_token_id in full_token_ids:
            think_pos = full_token_ids.index(think_token_id)
            result = full_token_ids[think_pos + 1:]
            return result
        else:
            return full_token_ids
        
    def _is_pad_or_bos_token_id(self, token_id: int) -> bool:
        if token_id == self.tokenizer.pad_token_id:
            return True
        if self.tokenizer.bos_token_id is not None and token_id == self.tokenizer.bos_token_id:
            return True
        return False

    def _remove_padding_tokens(self, token_id_list: List[int]):
        # Removes all the pad tokens or BOS tokens on the left-hand side using the 
        # pad token ID. This is more robust than looking for the string representation of
        # the pad token. Thus the prompt can begin with the literal string
        # "<|endoftext|>" (which is a common representation of the pad token).
        left_padding_removed = itertools.dropwhile(
            self._is_pad_or_bos_token_id, token_id_list
        )
        # Returns all tokens to the left of the first special token. This has
        # the effect of removing all right-hand padding. Moreover, it also
        # stops generation at other special tokens. For example, consider
        # StarCoder 2, where a completion may reach the end of a file and then
        # continue onto a second file: A<file_sep>B. The code below removes
        # <file_sep>B and only produces A.
        right_padding_removed = itertools.takewhile(
            lambda x: not self._is_pad_or_bos_token_id(x), left_padding_removed
        )

        return list(right_padding_removed)

    def decode_single_output(self, output_tensor, prompt=None):
        _ = prompt  # Suppress unused parameter warning
        full_token_ids = self._remove_padding_tokens(
            output_tensor.tolist()
        )

        # Capture raw completion before any processing
        raw_completion = self.tokenizer.decode(
            full_token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        
        # Remove prompt tokens to get only the generated response
        response_token_ids = self._remove_until_endthink_token(full_token_ids)
        
        # Filter out special tokens but keep content
        filtered_tokens = [
            token_id for token_id in response_token_ids 
            if not self._is_special_token_id(token_id)
        ]
        
        # Decode the response tokens
        response_text = self.tokenizer.decode(
            filtered_tokens,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )

        processed_completion = self._clean_code(response_text.strip()) 
        
        # Return both raw and processed completions
        return (processed_completion, raw_completion)

    def _clean_code(self, completion):
        """Clean up chat template completions to extract just the code"""
        import re

        # Extract code from markdown blocks if present
        code_block_match = re.search(r'```(?:\w+)?\s*\n(.*?)\n?```', completion, re.DOTALL)
        if code_block_match:
            extracted_code = code_block_match.group(1).strip()
            
            # Remove duplicate #lang racket if it already exists in prompt
            lines = extracted_code.split('\n')
            
            # Filter out duplicate #lang or other # directive lines
            filtered_lines = []
            seen_hash_directive = False

            while len(lines) > 0 and lines[0].strip() == '':
                lines = lines[1:]

            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.startswith('#lang'):
                    lines.pop(i)
                    break
            
            result = '\n'.join(lines)
            return result
        
        # Fallback: clean up the completion as-is
        lines = completion.split('\n')
        code_lines = []

        # Skip empty lines at the beginning
        start_idx = 0
        while start_idx < len(lines) and lines[start_idx].strip() == '':
            start_idx += 1
            
        # Collect non-empty lines
        for line in lines[start_idx:]:
            if line.strip():  # Only add non-empty lines
                code_lines.append(line)
        
        cleaned = '\n'.join(code_lines)
        result = cleaned if cleaned else completion
        return result
        

    def completions(
        self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, stop
    ):
        prompts = [prompt.strip() for prompt in prompts]
        output_tensors = self.completion_tensors(
            prompts,
            max_tokens,
            temperature,
            top_p,
        )


        results = []
        for (prompt, output_tensor) in zip(prompts, output_tensors):

            # Check if thinking process is incomplete
            completion_result = self._handle_thinking_budget(
                prompt, output_tensor, temperature, top_p, stop
            )

            results.append(completion_result)

        return results

    def _handle_thinking_budget(self, prompt, output_tensor, temperature, top_p, stop):
        """Handle incomplete thinking processes and regenerate if needed."""
        
        incomplete_status = self._is_incomplete(output_tensor)

        if incomplete_status == "Noend":
            # No end token found after </think>, regenerating

            # Get token IDs from tensor
            token_ids = output_tensor.tolist()

            # Convert </think> to token ID
            endthink_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

            # Find the position of </think>
            endthink_pos = token_ids.index(endthink_token_id)

            # Get tokens up to </think>
            tokens_before_endthink = token_ids[:endthink_pos + 1]

            output_tensor = torch.tensor(tokens_before_endthink, device="cuda")

            # Get the partial completion and add suffix
            processed_completion = self.tokenizer.decode(output_tensor, skip_special_tokens=False)
            recovered_prompt = processed_completion + "\n\n"

            # Regenerate with additional tokens
            additional_tokens = 512
            new_output_tensor = self.continue_completion_tensor(
                recovered_prompt,
                additional_tokens,
                temperature,
                top_p,
            )

            # Process the regenerated output
            new_tensor = new_output_tensor[0]

            # Combine original thinking + suffix + new completion
            final_processed_completion, final_raw_completion = self.decode_single_output(
                new_tensor, recovered_prompt
            )

            # Processing step

            # Apply stop token processing
            final_processed_completion = stop_at_stop_token(
                recovered_prompt + final_processed_completion, stop
            )

            return final_raw_completion
        
        elif incomplete_status == "Nothink":
            # No </think> token found, regenerating

            # Add thinking completion suffix
            thinking_suffix = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

            # Get the partial completion and add suffix
            processed_completion = self.tokenizer.decode(output_tensor, skip_special_tokens=False)
            recovered_prompt = processed_completion + thinking_suffix


            # Regenerate with additional tokens
            additional_tokens = 512
            new_output_tensor = self.continue_completion_tensor(
                recovered_prompt,
                additional_tokens,
                temperature,
                top_p,
            )

            # Process the regenerated output
            new_tensor = new_output_tensor[0]

            # Combine original thinking + suffix + new completion
            final_processed_completion, final_raw_completion = self.decode_single_output(
                new_tensor, recovered_prompt
            )

            return final_raw_completion

        else:
            # Normal processing path
            processed_completion, raw_completion = self.decode_single_output(output_tensor, prompt)
            final_processed_completion = stop_at_stop_token(processed_completion, stop)
            return raw_completion

    def _is_incomplete(self, output_tensor) -> str:
        """Check if the process is incomplete."""
        
        # Get token IDs from tensor
        token_ids = output_tensor.tolist()

        # Convert </think> to token ID
        endthink_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

        # Check if </think> token is missing
        if endthink_token_id not in token_ids:
            return "Nothink"

        # Find the position of </think>
        endthink_pos = token_ids.index(endthink_token_id)

        # Get tokens after </think>
        tokens_after_endthink = token_ids[endthink_pos + 1:]

        # Convert <|im_end|> to token ID
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Check if <|im_end|> token is missing after </think>
        if im_end_token_id not in tokens_after_endthink:
            return "Noend"

        return "Complete"

def automodel_partial_arg_parser():
    """
    This is also used by peftmodel.py.
    """
    args = partial_arg_parser()
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--revision", type=str)
    args.add_argument("--tokenizer_name", type=str)
    args.add_argument("--tokenizer_revision", type=str)
    args.add_argument("--name-override", type=str)
    args.add_argument("--flash-attention2", action="store_true")
    args.add_argument("--no-thinking", action="store_true",
                     help="Disable thinking mode (enabled by default)")
    return args


def do_name_override(args):
    """
    Applies the --name-override flag, or uses the model name, correcting / and - which the rest of
    the toolchain does not like.
    """
    if args.name_override:
        name = args.name_override
    else:
        name = args.name.replace("/", "_").replace("-", "_")
    return name


def main():
    args = automodel_partial_arg_parser()
    args = args.parse_args()
    model_kwargs = { }
    if args.flash_attention2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Configure thinking mode
    enable_thinking = not args.no_thinking

    model = Model(
        args.name, args.revision,
        model_kwargs=model_kwargs,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        enable_thinking=enable_thinking,
    )
    name = do_name_override(args)
    make_main(args, name, model.completions)


if __name__ == "__main__":
    main()
