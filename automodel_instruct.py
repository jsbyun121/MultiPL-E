from multipl_e.completions import make_main, partial_arg_parser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM
import itertools
from typing import List

lang2Language = {
    "r": "R",
    "rkt": "Racket",
    "ml": "OCaml",
    "jl": "Julia",
    "lua": "Lua",
}

class Model:
    def __init__(self, name, lang, use_chat_template=True, lora_path=None):
        if "qwen" in name.lower():
            dtype = torch.bfloat16
        elif "openai" in name.lower():
            dtype = "auto"
        else:
            raise ValueError(f"Unsupported model: {name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map="auto", trust_remote_code=True
        )
        if lora_path:
            print(f"Loading LoRA checkpoint from {lora_path}")
            self.model = PeftModelForCausalLM.from_pretrained(self.model, lora_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        assert (
            self.tokenizer.pad_token is not None
        ), "tokenizer has neither pad_token nor eos_token"

        self.use_chat_template = use_chat_template
        self.team_name = name.split("/")[0]
        self.model_name = name.split("/")[1]
        if self.model_name in ["Qwen3-4B-Thinking-2507", "gpt-oss-20b"]:
            self.enable_thinking = True
        else:
            self.enable_thinking = False

        self.language = lang2Language[lang.lower()]

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
                        {"role": "user", "content": f"Using given examples and the signature, generate the missing implementation in {self.language} by wrapping your code in ```{self.language.lower()} markdown blocks:\n\n{prompt}\n\n"}
                    ]
                    template_kwargs = {}
                elif self.team_name.lower() == "openai":
                    messages = [
                        {"role": "user", "content": f"Using given examples and the signature, generate the missing implementation in {self.language} by wrapping your code in ```{self.language.lower()} markdown blocks:\n\n{prompt}\n\n"}
                    ]
                    template_kwargs = {
                        "reasoning_effort": "medium"
                    }
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs
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
        left_padding_removed = itertools.dropwhile(
            self._is_pad_or_bos_token_id, token_id_list
        )
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
        post_completion = self.tokenizer.decode(
            self._remove_padding_tokens(full_token_ids[input_length:]),
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )

        return (pre_completion, post_completion)

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
        
        if self.enable_thinking:
            if self._is_thinking_complete(output_tensor):
                pre_completion, post_completion = self.decode_single_output(output_tensor, input_length)
                return (pre_completion, post_completion)
            else:
                if self.team_name.lower() == "qwen":
                    thinking_suffix = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                elif self.team_name.lower() == "openai":
                    thinking_suffix = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.<|end|><|start|>assistant<|channel|>final<|message|>"
                else:
                    raise ValueError(f"Unsupported model: {self.team_name}")

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
                pre_completion, post_completion = self.decode_single_output(
                    new_tensor, new_input_length
                )

                return (pre_completion, post_completion)
        else:
            # Non-thinking mode processing
            pre_completion, post_completion = self.decode_single_output(output_tensor, input_length)
            return (pre_completion, post_completion)

    def _is_thinking_complete(self, output_tensor) -> bool:
        """Check if thinking process is complete."""

        output_string = self.tokenizer.decode(output_tensor)

        if self.team_name.lower() == "qwen":
            if not "</think>" in output_string:
                return False
        elif self.team_name.lower() == "openai":
            if not "<|end|><|start|>assistant<|channel|>final<|message|>" in output_string:
                return False
        else:
            raise ValueError(f"Unsupported model: {self.team_name}")

        return True

def automodel_partial_arg_parser():
    """
    This is also used by peftmodel.py.
    """
    args = partial_arg_parser()
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--use-chat-template", action="store_true",
                        help="Use chat template for the model. This is useful for models that support chat templates, such as Qwen and OpenAI models.")
    args.add_argument("--lora-path", type=str, help="Path to the LORA checkpoint.")
    return args

def main():
    args = automodel_partial_arg_parser()
    args = args.parse_args()

    model = Model(
        args.name, args.lang,
        use_chat_template=args.use_chat_template,
        lora_path=args.lora_path,
    )

    model_name = args.name.replace("/", "_").replace("-", "_")
    make_main(args, model_name, model.completions)


if __name__ == "__main__":
    main()
