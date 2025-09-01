"""
This script produces completions for roughly any AutoModelForCausalLM with optional RAG support.
"""
from multipl_e.completions import make_main, partial_arg_parser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from typing import List, Optional, Tuple
from transformers import pipeline
from pprint import pprint

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts.prompt import PromptTemplate

class Model:
    def __init__(self, name, revision, model_kwargs, tokenizer_name=None, tokenizer_revision=None, 
                 use_chat_template=True, use_rag=False, vector_db=None, rag_prompt_template=None, rag_k=3):
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
        if self.model_name in ["Qwen3-4B-Thinking-2507", "gpt-oss-20b"]:
            self.enable_thinking = True
        else:
            self.enable_thinking = False

        # RAG components
        self.use_rag = use_rag
        self.vector_db = vector_db
        self.rag_prompt_template = rag_prompt_template
        self.rag_k = rag_k

        assert (
            len(self._all_special_token_ids) >= 1
        ), "tokenizer.all_special_ids() is empty"
        assert (
            self.tokenizer.pad_token_id in self._all_special_token_ids
        ), "pad_token_id not in all_special_ids"
        assert (
            self.tokenizer.eos_token_id in self._all_special_token_ids
        ), "eos_token_id not in all_special_ids"

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from the vector database."""
        if not self.use_rag or self.vector_db is None:
            return ""
        
        # Retrieve similar documents
        docs = self.vector_db.similarity_search(query, k=k)
        
        # Combine retrieved documents into context string
        context_str = "\n\n".join([doc.page_content for doc in docs])
        return context_str

    def augment_prompt_with_context(self, prompt: str, context: str) -> str:
        """Augment the prompt with retrieved context using the template."""
        if not context or not self.rag_prompt_template:
            return prompt
        
        # Use the RAG prompt template to format the augmented prompt
        augmented_prompt = self.rag_prompt_template.format(
            context_str=context,
            question=prompt
        )
        return augmented_prompt

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
        
        # Process prompts with proper chat template and RAG integration
        if self.use_chat_template and self.team_name.lower() in ["qwen", "openai"]:
            for prompt in prompts:
                # Retrieve RAG context if enabled
                rag_context = ""
                if self.use_rag:
                    rag_context = self.retrieve_context(prompt, k=self.rag_k)
                
                # Build the user message content
                if self.use_rag and rag_context:
                    # Integrate RAG context directly into the user message
                    user_content = f"""Using the following reference information:
{rag_context}

Based on the above information and given examples and the signature, generate the missing implementation by wrapping your code in ```language markdown blocks:

{prompt}
"""
                else:
                    # Original prompt without RAG
                    user_content = f"Using given examples and the signature, generate the missing implementation by wrapping your code in ```language markdown blocks:\n\n{prompt}\n\n"
                
                # Apply chat template based on model type
                if self.team_name.lower() == "qwen":
                    system_msg = "You are a helpful assistant."
                    if self.use_rag and rag_context:
                        system_msg = "You are a helpful assistant. Use the provided reference information to generate accurate code implementations."
                    
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                elif self.team_name.lower() == "openai":
                    messages = [
                        {"role": "user", "content": user_content}
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
            # Non-chat template path: apply RAG if enabled
            if self.use_rag:
                augmented_prompts = []
                for prompt in prompts:
                    context = self.retrieve_context(prompt)
                    augmented_prompt = self.augment_prompt_with_context(prompt, context)
                    augmented_prompts.append(augmented_prompt)
                final_prompts = augmented_prompts
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


def initialize_rag_components(doc_path: str, embedding_model_name: str) -> Tuple[Optional[FAISS], Optional[PromptTemplate]]:
    """Initialize RAG components including vector database and prompt template."""
    if not doc_path:
        return None, None
    
    try:
        # Load documents
        loader = PyMuPDFLoader(doc_path)

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Split documents into chunks
        splitter = SemanticChunker(
            embeddings = embeddings,
            number_of_chunks=4
        )
        chunked_docs = splitter.split_documents(loader.load())
        
        # Create embeddings and vector database
        
        db = FAISS.from_documents(chunked_docs, embeddings)
        
        # Define RAG prompt template
        PROMPT_TEMPLATE = """Known information:
{context_str}

Based on the above known information, respond to the user's question concisely and professionally. If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided,' and do not include fabricated details in the answer. Please respond in English.

The question is: {question}"""
        
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, 
            input_variables=["context_str", "question"]
        )
        
        print(f"Successfully initialized RAG with {len(chunked_docs)} document chunks")
        return db, prompt
        
    except Exception as e:
        print(f"Error initializing RAG components: {e}")
        return None, None


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
    args.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                      help="HuggingFace embedding model to use for RAG")
    args.add_argument("--doc-path", type=str,
                      help="Path to PDF document for RAG")
    args.add_argument("--use-rag", action="store_true",
                      help="Enable RAG (Retrieval-Augmented Generation) mode")
    args.add_argument("--use-chat-template", action="store_true",
                      help="Use chat template for the model. This is useful for models that support chat templates, such as Qwen and OpenAI models.")
    args.add_argument("--rag-k", type=int, default=3,
                      help="Number of documents to retrieve for RAG (default: 3)")
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
    
    model_kwargs = {}
    if args.flash_attention2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Initialize RAG components if enabled
    vector_db = None
    rag_prompt_template = None
    
    if args.use_rag:
        if not args.doc_path:
            print("Warning: --use-rag is enabled but --doc-path is not provided. RAG will be disabled.")
            args.use_rag = False
        else:
            print(f"Initializing RAG with document: {args.doc_path}")
            vector_db, rag_prompt_template = initialize_rag_components(
                args.doc_path, 
                args.embedding_model
            )
            if vector_db is None:
                print("Warning: Failed to initialize RAG components. Continuing without RAG.")
                args.use_rag = False
    
    # Initialize model with RAG components
    model = Model(
        args.name, 
        args.revision,
        model_kwargs=model_kwargs,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        use_chat_template=args.use_chat_template,
        use_rag=args.use_rag,
        vector_db=vector_db,
        rag_prompt_template=rag_prompt_template,
        rag_k=args.rag_k if hasattr(args, 'rag_k') else 3  # Default to 3 if rag_k is not set
    )
    
    name = do_name_override(args)
    
    # Print configuration summary
    print(f"Model initialized: {args.name}")
    print(f"RAG enabled: {args.use_rag}")
    if args.use_rag:
        print(f"Embedding model: {args.embedding_model}")
        print(f"Document path: {args.doc_path}")
        print(f"Retrieval k: {args.rag_k}")
    
    make_main(args, name, model.completions)


if __name__ == "__main__":
    main()