"""
This script produces completions for roughly any AutoModelForCausalLM.
"""

# Standard library imports
import itertools
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# LangChain imports
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Local imports
from multipl_e.completions_tool import make_main, partial_arg_parser
from rag_queries import QUERIES
from rag_examples import QUERY_JL, DOC_JL, QUERY_ML, DOC_ML, QUERY_LUA, DOC_LUA, QUERY_R, DOC_R, QUERY_RKT, DOC_RKT

# Use imported queries instead of hardcoded list
queries = QUERIES

def query_documentation_pair(lang: str) -> str:
    if lang == "jl":
        return f"Query: {QUERY_JL}\n\nRetrieved Documentation: {DOC_JL}"
    elif lang == 'ml':
        return f"Query: {QUERY_ML}\n\nRetrieved Documentation: {DOC_ML}"
    elif lang == 'lua':
        return f"Query: {QUERY_LUA}\n\nRetrieved Documentation: {DOC_LUA}"
    elif lang == 'r':
        return f"Query: {QUERY_R}\n\nRetrieved Documentation: {DOC_R}"
    elif lang == 'rkt':
        return f"Query: {QUERY_RKT}\n\nRetrieved Documentation: {DOC_RKT}"
    else:
        raise ValueError(f"Unsupported language for query-documentation pair: {lang}")
    
@dataclass
class CompletionResult:
    """Result of multi-generation completion."""
    intermediate_prompt: str  # Chat prompt for query generation
    intermediate_response: str  # Model's response with the query
    final_prompt: str  # Chat prompt for code generation
    final_response: str  # Model's final code response
    generation_count: int  # Number of generation steps
    success: bool

class RAGTools:
    """Enhanced RAG tools manager for document retrieval with caching."""
    
    def __init__(self, 
                 embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 lang: Optional[str] = None,
                 k: int = 3):
        self.embedding_model = embedding_model
        self.k = k
        self.db = None
        self.retrieval_cache = {}
        self.lang = lang
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the vector database from documents."""
        if not self.lang:
            print("Warning: No language specified for RAG initialization")
            self.db = None
            return None
            
        default_base_dir = "/storage/junsoo/api_json"
        base_dir = os.environ.get("API_JSON_PATH", default_base_dir)
        file_path = f"{base_dir}/{self.lang}/{self.lang}_api.json"

        print(f"Initializing RAG with documents from {file_path}")

        try:
            loader = JSONLoader(
                file_path,
                '.text',
                text_content=False,
                json_lines=True,
            )

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            docs = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            vector_db = FAISS.from_documents(
                docs,
                embeddings
            )

            self.db = vector_db
            print(f"Successfully initialized RAG with {len(docs)} document chunks")
            return vector_db

        except Exception as e:
            print(f"Error initializing RAG tools: {str(e)}")
            self.db = None
            return None

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context from the document database with caching.
        
        Args:
            query: The search query to find relevant information
            
        Returns:
            Retrieved context as a formatted string
        """
        # Check cache first for efficiency
        if query in self.retrieval_cache:
            return self.retrieval_cache[query]
            
        if not self.db:
            result = "No documents available for retrieval."
            self.retrieval_cache[query] = result
            return result
        
        try:
            # Retrieve similar documents with scores
            docs_with_scores = self.db.similarity_search_with_score(query, k=self.k)
            
            if not docs_with_scores:
                result = "No relevant context found for the query."
            else:
                context_str = "\n\n".join([doc.page_content for (doc, _) in docs_with_scores])
                result = f"Retrieved Documentation:\n{context_str}"
            
            # Cache the result
            self.retrieval_cache[query] = result
            return result
        
        except Exception as e:
            result = f"Error retrieving context: {str(e)}"
            self.retrieval_cache[query] = result
            return result


class MultiGenerationEngine:
    """Two-stage prompt-based RAG-enhanced code generation."""
    
    def __init__(self, model, rag_tools: Optional[RAGTools] = None):
        self.model = model
        self.rag_tools = rag_tools
        self.model_name = model.model_name
    
    def _extract_query_from_response(self, response: str) -> Optional[str]:
        """Extract query from Stage 1 response using JSON format."""
        import json
        
        # First try to find JSON in the response
        json_pattern = r'\{[^}]*"query"[^}]*\}'
        json_match = re.search(json_pattern, response, re.IGNORECASE | re.DOTALL)
        if json_match:
            try:
                json_obj = json.loads(json_match.group(0))
                if "query" in json_obj:
                    return json_obj["query"].strip()
            except json.JSONDecodeError:
                pass
        
        return None

    def generate_batch_two_stages(self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, lang: str = "python", force_choice: bool = False) -> List[CompletionResult]:
        """Generate completions for multiple prompts using batch two-stage process."""
        print(f"🔍 Batch Stage 1: Generating queries for {len(prompts)} prompts...")
        # print(f"DEBUG: force_choice in generate_batch_two_stages = {force_choice}")

        # Stage 1: Batch generate RAG queries
        stage1_results = self._batch_stage1_generate_queries(prompts, max_tokens, temperature, top_p, force_choice, lang)
        
        # Extract queries and retrieve contexts
        retrieved_contexts = []
        for i, result in enumerate(stage1_results):
            query = result.get("query")
            if query and self.rag_tools:
                print(f"📚 Prompt {i+1}: Retrieved query '{query}'")
                context = self.rag_tools.retrieve_context(query)
                retrieved_contexts.append(context)
            else:
                retrieved_contexts.append("")
        
        # Stage 2: Batch generate code with retrieved contexts
        print(f"💻 Batch Stage 2: Generating code for {len(prompts)} prompts...")
        stage2_results = self._batch_stage2_generate_code(prompts, retrieved_contexts, max_tokens, temperature, top_p, lang)
        
        # Combine results
        completion_results = []
        for i in range(len(prompts)):
            result = CompletionResult(
                intermediate_prompt=stage1_results[i]["chat_prompt"],
                intermediate_response=stage1_results[i]["response"],
                final_prompt=stage2_results[i]["chat_prompt"],
                final_response=stage2_results[i]["response"],
                generation_count=2,
                success=stage2_results[i]["response"] is not None
            )
            completion_results.append(result)
        
        return completion_results
    
    def _batch_stage1_generate_queries(self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, force_choice: bool, lang: str) -> List[Dict[str, Any]]:
        """Batch Stage 1: Generate queries for RAG retrieval for multiple prompts."""
        batch_messages = []
        # print(f"DEBUG: force_choice in _batch_stage1_generate_queries = {force_choice}")

        for prompt in prompts:
            if force_choice:   
                user_content = f"""{prompt}

Based on this coding task, what specific information do you need from the documentation? Select a focused search query from the following list that would help you find relevant API documentation, syntax details, or examples:

{queries}

Output format (JSON):
{{
  "query": "your specific search query here",
  "reasoning": "brief explanation of why this query is relevant"
}}"""
            
            else:
                user_content = f"""{prompt}

Based on this coding task, what specific information do you need from the documentation? Formulate a precise search query based on a following example that would help you find relevant API documentation, syntax details, or examples:

{query_documentation_pair(lang)}

Output format (JSON):
{{
  "query": "your specific search query here",
  "reasoning": "brief explanation of why this query is relevant"
}}"""
            
            messages = [
                {"role": "user", "content": user_content}
            ]
            batch_messages.append(messages)
        
        # Convert messages to text format for batch processing
        batch_texts = []
        for messages in batch_messages:
            text = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_texts.append(text)
        
        # Tokenize all texts together
        inputs = self.model.tokenizer(
            batch_texts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True
        ).to("cuda")
        
        # Generate batch responses
        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_tokens,  # Limit tokens for query generation
                pad_token_id=self.model.tokenizer.pad_token_id
            )
        
        # Decode and process responses
        results = []
        for i, output in enumerate(outputs):
            chat_prompt_tokens = self.model._remove_padding_tokens(output[:len(inputs["input_ids"][i])].tolist())
            response_tokens = self.model._remove_padding_tokens(output[len(inputs["input_ids"][i]):].tolist())
            
            chat_prompt = self.model.tokenizer.decode(
                chat_prompt_tokens,
                skip_special_tokens=False
            ).strip()

            response = self.model.tokenizer.decode(
                response_tokens,
                skip_special_tokens=False
            ).strip()
            
            query = self._extract_query_from_response(response)

            results.append({
                "chat_prompt": chat_prompt,
                "response": response,
                "query": query
            })
        
        return results
    
    def _batch_stage2_generate_code(self, prompts: List[str], contexts: List[str], max_tokens: int, temperature: float, top_p: float, lang: str) -> List[Dict[str, Any]]:
        """Batch Stage 2: Generate code for multiple prompts with retrieved contexts."""
        batch_messages = []
        
        for i, (prompt, context) in enumerate(zip(prompts, contexts)):
            if context and "Retrieved Documentation:" in context:
                user_content = f"""{context}

Original Task:
{prompt}

Based on the above documentation and examples, generate the missing implementation in {lang} by wrapping your code in ```{lang} markdown blocks."""
            else:
                user_content = f"""{prompt}

Generate the missing implementation in {lang} by wrapping your code in ```{lang} markdown blocks."""

            messages = [
                {"role": "user", "content": user_content}
            ]
            batch_messages.append(messages)
        
        # Convert messages to text format for batch processing
        batch_texts = []
        for messages in batch_messages:
            text = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_texts.append(text)
        
        # Tokenize all texts together
        inputs = self.model.tokenizer(
            batch_texts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True
        ).to("cuda")
        
        # Generate batch responses
        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_tokens,
                pad_token_id=self.model.tokenizer.pad_token_id
            )
        
        # Decode and process responses
        results = []
        for i, output in enumerate(outputs):
            chat_prompt_tokens = self.model._remove_padding_tokens(output[:len(inputs["input_ids"][i])].tolist())
            response_tokens = self.model._remove_padding_tokens(output[len(inputs["input_ids"][i]):].tolist())
            
            chat_prompt = self.model.tokenizer.decode(
                chat_prompt_tokens,
                skip_special_tokens=False
            ).strip()

            response = self.model.tokenizer.decode(
                response_tokens,
                skip_special_tokens=False
            ).strip()

            results.append({
                "chat_prompt": chat_prompt,
                "response": response
            })
        
        return results
    


class Model:
    def __init__(self, name, revision, model_kwargs, tokenizer_name=None, 
                 tokenizer_revision=None, use_chat_template=True, rag_tools=None, use_rag=False, lang="python"):
        # Determine dtype based on model
        if "qwen" in name.lower():
            dtype = torch.bfloat16
        elif "openai" in name.lower():
            dtype = "auto"
        else:
            dtype = torch.float16  # Default fallback
        
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype, device_map="auto", 
            trust_remote_code=True, **model_kwargs
        ).cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or name,
            revision=tokenizer_revision,
            padding_side="left",
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        assert self.tokenizer.pad_token is not None, "tokenizer has neither pad_token nor eos_token"

        self._all_special_token_ids = self.tokenizer.all_special_ids
        self.use_chat_template = use_chat_template
        self.model_name = name.split("/")[1] if "/" in name else name
        
        self.enable_thinking = self.model_name in ["Qwen3-4B-Thinking-2507", "gpt-oss-20b"]
        
        self.use_rag = use_rag
        self.saved_context = ""
        self.lang = lang
        
        # Initialize multi-generation components
        self.multi_gen_engine = MultiGenerationEngine(self, rag_tools)

    def continue_completion_tensor(self, prompts: List[str], max_tokens: int, 
                                 temperature: float, top_p: float):
        """Continue generation from a prompt."""
        self.model.eval()
        
        inputs = self.tokenizer(
            prompts,
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
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output, inputs['input_ids']

    def completion_tensors(self, prompts: list, max_tokens: int, 
                          temperature: float, top_p: float):
        """Generate completions for multiple prompts."""
        self.model.eval()
        formatted_prompts = []
        
        # Format prompts with chat template if needed
        if self.use_chat_template:
            for prompt in prompts:
                messages = self._create_messages(prompt)

                # Apply chat template
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True
                }

                # Add model-specific parameters
                if "gpt-oss" in self.model_name.lower() and self.enable_thinking:
                    template_kwargs["reasoning_effort"] = "medium"
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs
                )
                formatted_prompts.append(text)

            final_prompts = formatted_prompts
        else:
            final_prompts = prompts

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
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return output_tensors, inputs['input_ids']

    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Create messages for chat template."""
        if self.saved_context:
            user_message = f"{self.saved_context}\n\n{prompt}\n\nBased on the above context, generate the complete implementation:"
        else:
            user_message = f"{prompt}\n\nGenerate the complete implementation:"

        messages = [
            {"role": "user", "content": user_message}
        ]
        
        return messages

    def _is_pad_or_bos_token_id(self, token_id: int) -> bool:
        """Check if token is padding or BOS."""
        if token_id == self.tokenizer.pad_token_id:
            return True
        if self.tokenizer.bos_token_id is not None and token_id == self.tokenizer.bos_token_id:
            return True
        return False

    def _remove_padding_tokens(self, token_id_list: List[int]):
        """Remove padding tokens from token list."""
        left_padding_removed = itertools.dropwhile(
            self._is_pad_or_bos_token_id, token_id_list
        )
        right_padding_removed = itertools.takewhile(
            lambda x: not self._is_pad_or_bos_token_id(x), left_padding_removed
        )
        return list(right_padding_removed)

    def decode_single_output(self, output_tensor, input_length):
        """Decode a single output tensor."""
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
        self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, stop, force_choice: bool
    ):   
        prompts = [prompt.strip() for prompt in prompts]
        # print(f"DEBUG: force_choice in completions = {force_choice}")

        # Use two-stage RAG if enabled, otherwise fallback to original logic
        if self.use_rag and hasattr(self, 'multi_gen_engine'):
            return self._two_stage_completions(prompts, max_tokens, temperature, top_p, stop, force_choice)
        else:
            return self._original_completions(prompts, max_tokens, temperature, top_p, stop)

    def _two_stage_completions(self, prompts: List[str], max_tokens: int,
                              temperature: float, top_p: float, stop, force_choice: bool) -> List[Dict[str, Any]]:
        """Batch two-stage prompt-based RAG completion."""
        # print(f"DEBUG: force_choice in _two_stage_completions = {force_choice}")
        try:
            # Use batch processing for all prompts at once 
            completion_results = self.multi_gen_engine.generate_batch_two_stages(
                prompts, max_tokens, temperature, top_p, self.lang, force_choice
            )
            
            # Format results with new field names
            formatted_results = []
            for completion_result in completion_results:
                result = {
                    "intermediate_prompt": completion_result.intermediate_prompt,
                    "intermediate_response": completion_result.intermediate_response,
                    "final_prompt": completion_result.final_prompt,
                    "final_response": completion_result.final_response,
                    "success": completion_result.success
                }
                formatted_results.append(result)
            
            return formatted_results
                
        except Exception as e:
            # Handle batch errors gracefully - return error for all prompts
            print(f"Batch processing error: {str(e)}")
            results = []
            for _ in prompts:
                result = {
                    "intermediate_prompt": "",
                    "intermediate_response": f"Batch error during generation: {str(e)}",
                    "final_prompt": "",
                    "final_response": "",
                    "success": False
                }
                results.append(result)
            return results
    
    def _original_completions(self, prompts: List[str], max_tokens: int, 
                            temperature: float, top_p: float, stop) -> List[Dict[str, Any]]:
        """Original completion logic for backward compatibility."""
        output_tensors, input_ids = self.completion_tensors(
            prompts,
            max_tokens,
            temperature,
            top_p,
        )

        results = []
        
        # Process each output tensor
        for i, (output_tensor, input_length) in enumerate(zip(output_tensors, input_ids)):
            try:
                # Handle thinking budget if enabled
                if self.enable_thinking:
                    pre_completion, post_completion = self._handle_thinking_budget(
                        prompts[i], output_tensor, len(input_length), temperature, top_p, stop
                    )
                else:
                    pre_completion, post_completion = self.decode_single_output(
                        output_tensor, len(input_length)
                    )
                
                result = {
                    "intermediate_prompt": "",  # No intermediate prompt in non-RAG mode
                    "intermediate_response": pre_completion,  # Map pre_completion to intermediate_response  
                    "final_prompt": prompts[i],  # Original prompt becomes final_prompt
                    "final_response": post_completion,  # Map post_completion to final_response
                    "success": True
                }
                results.append(result)
                
            except Exception as e:
                result = {
                    "intermediate_prompt": "",
                    "intermediate_response": f"Error: {str(e)}",
                    "final_prompt": prompts[i],
                    "final_response": "",
                    "success": False
                }
                results.append(result)
        
        return results

    def _handle_thinking_budget(self, prompt: str, output_tensor: torch.Tensor, input_length: int, 
                               temperature: float, top_p: float, stop) -> Tuple[str, str]:
        """Handle incomplete thinking processes and regenerate if needed."""
        if self._is_thinking_complete(output_tensor):
            return self.decode_single_output(output_tensor, input_length)
        
        # Add thinking completion suffix based on model
        if "qwen3" in self.model_name.lower():
            thinking_suffix = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
        elif "gpt-oss" in self.model_name.lower():
            thinking_suffix = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.<|end|><|start|>assistant<|channel|>final<|message|>"
        else:
            thinking_suffix = "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n\n"

        processed_completion = self.tokenizer.decode(
            output_tensor, skip_special_tokens=False
        )
        recovered_prompt = processed_completion + thinking_suffix

        # Regenerate with additional tokens
        new_output_tensor, new_input_ids = self.continue_completion_tensor(
            [recovered_prompt], 512, temperature, top_p
        )

        return self.decode_single_output(
            new_output_tensor[0], new_input_ids.shape[1]
        )

    def _is_thinking_complete(self, output_tensor) -> bool:
        """Check if thinking process is complete."""
        output_string = self.tokenizer.decode(output_tensor)

        if "qwen3" in self.model_name.lower():
            return "</think>" in output_string
        elif "gpt-oss" in self.model_name.lower():
            return "<|end|><|start|>assistant<|channel|>final<|message|>" in output_string
        
        return True  # Default: assume complete


def automodel_partial_arg_parser():
    """Argument parser for the script."""
    args = partial_arg_parser()
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--revision", type=str)
    args.add_argument("--tokenizer_name", type=str)
    args.add_argument("--tokenizer_revision", type=str)
    args.add_argument("--name-override", type=str)
    args.add_argument("--flash-attention2", action="store_true")
    args.add_argument("--use-chat-template", action="store_true",
                     help="Use chat template for the model.")
    
    # RAG-specific arguments
    args.add_argument("--use-rag", action="store_true",
                     help="Enable RAG tool functionality")
    args.add_argument("--force-choice", action="store_true",
                     help="Force choosing from predefined queries")
    args.add_argument("--embedding-model", type=str, 
                     default="Qwen/Qwen3-Embedding-0.6B",
                     help="Embedding model for RAG")
    args.add_argument("--k", type=int, default=3,
                     help="Number of documents to retrieve")
    return args


def do_name_override(args):
    """Apply name override or format the model name."""
    if args.name_override:
        return args.name_override
    return args.name.replace("/", "_").replace("-", "_")


def main():
    args = automodel_partial_arg_parser()
    args = args.parse_args()
    model_kwargs = {}
    
    if args.flash_attention2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Initialize RAG tools if requested
    rag_tools = None
    if args.use_rag:
        if args.lang:
            rag_tools = RAGTools(
                embedding_model=args.embedding_model,
                lang=args.lang,
                k=args.k
            )
            print(f"RAG tools initialized for language '{args.lang}'")
        else:
            print("Warning: --use-rag specified but no language provided with --lang. RAG functionality disabled.")
            args.use_rag = False

    model = Model(
        args.name, 
        args.revision,
        model_kwargs=model_kwargs,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        use_chat_template=args.use_chat_template,
        use_rag=args.use_rag,
        rag_tools=rag_tools,
        lang=args.lang if args.lang else "python"
    )

    name = do_name_override(args)
    make_main(args, name, model.completions)


if __name__ == "__main__":
    main()