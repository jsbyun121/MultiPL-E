"""
This script produces completions for roughly any AutoModelForCausalLM.
"""
from multipl_e.completions import make_main, partial_arg_parser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from typing import List
from transformers import pipeline

class Model:
    def __init__(self, name, revision, model_kwargs, tokenizer_name=None, tokenizer_revision=None,  use_chat_template=True):
        if "qwen" in name.lower():
            dtype = torch.bfloat16
        elif "openai" in name.lower():
            dtype = "auto"
        else:
            raise ValueError(f"Unsupported model: {name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype, device_map="auto", trust_remote_code=True, **model_kwargs
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
        self.team_name = name.split("/")[0]
        self.model_name = name.split("/")[1]
        self.enable_thinking = "think" in self.model_name.lower()

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
        self.model.eval()  # Set model to evaluation mode
        
        inputs = self.tokenizer(
            [prompt],
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
        ).to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output, inputs['input_ids']


    def completion_tensors(
        self,
        prompts: list,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.model.eval() # Not essential, but just in case.
        formatted_prompts = []
        # If using chat template, format prompts accordingly
        if self.use_chat_template and self.team_name.lower() in ["qwen", "openai"]:
            for prompt in prompts:
                if self.team_name.lower() == "qwen":
                    system_msg = "You are a helpful assistant."
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"Using given examples and the signature, generate the missing implementation by wrapping your code in ```language markdown blocks:\n\n{prompt}\n\n"}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                elif self.team_name.lower() == "openai":
                    messages = [
                        {"role": "user", "content": f"Using given examples and the signature, generate the missing implementation by wrapping your code in ```language markdown blocks:\n\n{prompt}\n\n"}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        reasoning_effort="medium"
                    )
                formatted_prompts.append(text)
            final_prompts = formatted_prompts
        else:
            final_prompts = prompts

        # Tokenize the prompts
        inputs = self.tokenizer(
            final_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
        ).to("cuda")

        with torch.no_grad():
            output_tensors = self.model.generate(
                **inputs,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Return the generated tensors along with the original 'input_ids' to use as a separation criterion.
        return output_tensors, inputs['input_ids']

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

    def decode_single_output(self, output_tensor, input_length):
        full_token_ids = output_tensor.tolist()

        pre_completion = self.tokenizer.decode(
            self._remove_padding_tokens(full_token_ids[:input_length]),
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        raw_completion = self.tokenizer.decode(
            self._remove_padding_tokens(full_token_ids[input_length:]),
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )

        return (pre_completion, raw_completion)

    def completions(
        self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, stop
    ):
        prompts = [prompt.strip() for prompt in prompts]
        output_tensors, input_ids = self.completion_tensors(
            prompts,
            max_tokens,
            temperature,
            top_p,
        )

        input_length = input_ids.shape[1]

        results = []
        for (prompt, output_tensor) in zip(prompts, output_tensors):

            # Check if thinking process is incomplete
            pre_completion, completion_result = self._handle_thinking_budget(
                prompt, output_tensor, input_length, temperature, top_p, stop
            )

            results.append((pre_completion, completion_result))

        return results

    def _handle_thinking_budget(self, prompt, output_tensor, input_length, temperature, top_p, stop):
        """Handle incomplete thinking processes and regenerate if needed."""
        
        complete_status = self._is_complete(output_tensor)

        if self.enable_thinking:

            if complete_status:
                pre_completion, raw_completion = self.decode_single_output(output_tensor, input_length)
                return (pre_completion, raw_completion)

            else:
                thinking_suffix = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

                # Get the partial completion and add suffix
                processed_completion = self.tokenizer.decode(output_tensor, skip_special_tokens=False)
                recovered_prompt = processed_completion + thinking_suffix

                # Regenerate with additional tokens
                additional_tokens = 512
                new_output_tensor, new_input_ids = self.continue_completion_tensor(
                    recovered_prompt,
                    additional_tokens,
                    temperature,
                    top_p,
                )

                # Process the regenerated output
                new_tensor = new_output_tensor[0]
                new_input_length = new_input_ids.shape[1]


                # Combine original thinking + suffix + new completion
                pre_completion, raw_completion = self.decode_single_output(
                    new_tensor, new_input_length
                )

                return (pre_completion, raw_completion)
                
            
        else:
            # Non-thinking mode processing
            pre_completion, raw_completion = self.decode_single_output(output_tensor, input_length)
            return (pre_completion, raw_completion)

    def _is_complete(self, output_tensor) -> bool:
        """Check if the process is complete."""

        # Get token IDs from tensor
        token_ids = output_tensor.tolist()

        if self.team_name.lower() == "qwen":
            # Convert </think> to token ID
            endthink_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        elif self.team_name.lower() == "openai":
            raise NotImplementedError("Thinking mode is not supported for OpenAI models.")
        else:
            raise ValueError(f"Unsupported model: {self.team_name}")

        # Check if </think> token is missing
        if endthink_token_id not in token_ids:
            return False

        return True

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
    args.add_argument("--use-chat-template", action="store_true",
                        help="Use chat template for the model. This is useful for models that support chat templates, such as Qwen and OpenAI models.")
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

    model = Model(
        args.name, args.revision,
        model_kwargs=model_kwargs,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        use_chat_template=args.use_chat_template,
    )

    name = do_name_override(args)
    make_main(args, name, model.completions)


if __name__ == "__main__":
    main()
