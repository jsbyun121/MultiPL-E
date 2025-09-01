"""
This script produces completions for roughly any AutoModelForCausalLM with RAG tool support.
Enhanced with iterative multi-tool calling with a 3-call limit.
"""
from multipl_e.completions_tool import make_main, partial_arg_parser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import re
import ast
from dataclasses import dataclass

from langchain_core.tools import tool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import UnstructuredPDFLoader

@dataclass
class CompletionResult:
    """Result of multi-generation completion."""
    pre_completion: List[str]  # All processing steps
    post_completion: Optional[str]  # Final code result
    tool_calls_made: List[Dict[str, Any]]
    context_accumulated: str
    generation_count: int
    success: bool

class RAGQueryInput(BaseModel):
    """Input schema for RAG query tool."""
    query: str = Field(description="The search query to find relevant information from documents")

class CodeGenerationInput(BaseModel):
    """Input schema for code generation tool."""
    code: str = Field(description="The generated executable code implementation")

class RAGTools:
    """Enhanced RAG tools manager for document retrieval with caching."""
    
    def __init__(self, doc_path: Optional[str] = None, 
                 embedding_model: str = "Qwen/Qwen3-Embedding-0.6B", 
                 k: int = 3):
        self.doc_path = doc_path
        self.embedding_model = embedding_model
        self.k = k
        self.db = None
        self.retrieval_cache = {}
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the vector database from documents."""
        try:
            loader = UnstructuredPDFLoader(
                self.doc_path

            )
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_amount=90.0
            )
            docs = loader.load()
            chunked_docs = splitter.split_documents(docs)
            self.db = FAISS.from_documents(chunked_docs, embeddings)
        except Exception as e:
            print(f"Warning: Could not initialize RAG database: {e}")
            self.db = None

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
    
    def get_tools(self) -> List[callable]:
        """Get the RAG tools as callable functions for chat template."""
        def retrieve_documents(query: str) -> str:
            """Retrieve relevant documentation for code implementation.
            
            Args:
                query: The search query to find relevant information from API documentation
            Returns:
                Retrieved documentation content as a string
            """
            return self.retrieve_context(query)
            
        return [retrieve_documents]


class CodeGenerationTool:
    """Tool for generating code."""
    
    def get_tools(self) -> List[callable]:
        """Get code generation tools as callable functions for chat template."""
        def generate_code(code: str) -> str:
            """Generate final code implementation.
            
            Args:
                code: The generated executable code implementation
            Returns:
                Wrapped code block to be parsed and verified for code accuracy
            """
            
            return "<code>" + code + "</code>"
        return [generate_code]

    def extract_code(self, text: str) -> str:
        """Extract code from generated text."""
        # Look for <code> tags first
        code_tag_pattern = r'<code>(.*?)</code>'
        matches = re.findall(code_tag_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        # Fallback: return the original text
        return text.strip()


class MultiGenerationEngine:
    """Multi-generation engine with tool selection for RAG-enhanced code generation."""
    
    def __init__(self, model, rag_tools: Optional[RAGTools] = None, max_iterations: int = 3):
        self.model = model
        self.rag_tools = rag_tools
        self.code_tool = CodeGenerationTool()
        self.max_iterations = max_iterations
        
        # Create combined tool list
        self.available_tools = []
        if rag_tools:
            self.available_tools.extend(rag_tools.get_tools())
        self.available_tools.extend(self.code_tool.get_tools())

        self.model_name = model.model_name
    
    def _extract_code_from_result(self, tool_result: str) -> str:
        """Extract code from tool result."""
        return self.code_tool.extract_code(tool_result)

    def parse_tool_call(self, output_text: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from model output (supports OpenAI OSS + Qwen3)."""

        # Pattern 1: gpt-oss style
        if "gpt-oss" in self.model_name.lower():
            match = re.search(
                r"<\|channel\|>commentary to=functions\.([^\s]+)\s*<\|constrain\|>json<\|message\|>\s*({.*?})\s*<\|call\|>",
                output_text,
                re.DOTALL,
            )
            if match:
                func_name = match.group(1)
                args_json = match.group(2)
                try:
                    args = json.loads(args_json)
                    return {"name": func_name, "arguments": args}
                except json.JSONDecodeError:
                    return None

        # Pattern 2: Qwen3 style
        elif "qwen3" in self.model_name.lower():
            match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", output_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None

        else:
            raise NotImplementedError("Tool call parsing not implemented for this model.")

    def execute_tool(self, tool_call: Dict[str, Any]) -> Tuple[str, str]:
        """Execute a tool call (normalized schema)."""

        # Normalize keys
        tool_name = tool_call.get("name")
        parameters = tool_call.get("arguments", {})

        # Map tool names to functions
        if tool_name == "retrieve_documents" and self.rag_tools:
            query = parameters.get("query", "")
            return (query, self.rag_tools.retrieve_context(query))
        elif tool_name == "generate_code":
            code = parameters.get("code", "")
            return ("", self.code_tool.get_tools()[0](code))
        else:
            raise ValueError(f"Unknown tool call: {tool_name}")
    
    def generate_with_tools(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> CompletionResult:
        """Generate completion with multi-step tool usage."""
        pre_completion = []
        post_completion = None
        tool_calls_made = []
        context_accumulated = ""
        current_prompt = prompt
        
        for iteration in range(self.max_iterations):
            # Generate response with available tools
            messages = self._create_tool_messages(current_prompt, context_accumulated, iteration)
            
            # Apply chat template with tools
            inputs = self.model.tokenizer.apply_chat_template(
                messages, 
                tools=self.available_tools,
                add_generation_prompt=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to("cuda")
            
            # Generate response
            with torch.no_grad():
                output = self.model.model.generate(
                    **inputs,
                    do_sample=True,
                    use_cache=True,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.model.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.model.tokenizer.decode(
                output[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=False
            )
            
            pre_completion.append(response)
            
            # Check for tool call
            tool_call = self.parse_tool_call(response)
            
            if tool_call:
                tool_calls_made.append(tool_call)
                query, tool_result = self.execute_tool(tool_call)
                
                # If this is generate_code tool, we're done
                if tool_call.get("name") == "generate_code":
                    # Extract code from the tool result instead of parameters
                    post_completion = self._extract_code_from_result(tool_result)
                    break
                else:
                    # Continue with accumulated context
                    context_accumulated += f"\n\nDocumenation Snippet about {query}:\n\n{tool_result}"
                    current_prompt = f"{prompt}\n\nPrevious context: {context_accumulated}"
            else:
                # No tool call detected, assume completion
                post_completion = self.code_tool.extract_code(response)
                break
        
        return CompletionResult(
            pre_completion=pre_completion,
            post_completion=post_completion,
            tool_calls_made=tool_calls_made,
            context_accumulated=context_accumulated,
            generation_count=len(pre_completion),
            success=post_completion is not None
        )
    
    def _create_tool_messages(self, prompt: str, context: str, iteration: int) -> List[Dict[str, str]]:
        """Create messages for tool-enabled chat template."""
        system_content = "You are a helpful coding assistant."
        
        if iteration == 0:
            user_content = f"{prompt}\n\nFirst, determine what additional information you need and use retrieve_documents if needed, then generate_code with your implementation."
        else:
            user_content = f"{prompt}\n\nContext so far: {context}\n\nContinue with the next step or use generate_code if ready."
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]



class Model:
    def __init__(self, name, revision, model_kwargs, tokenizer_name=None, 
                 tokenizer_revision=None, use_chat_template=True, rag_tools=None, max_tool_calls=3, use_rag=False):
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
        self.available_tool_calls = max_tool_calls
        
        # Initialize multi-generation components
        self.multi_gen_engine = MultiGenerationEngine(self, rag_tools, max_tool_calls)

    def continue_completion_tensor(self, prompts: List[str], max_new_tokens: int, 
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
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output, inputs['input_ids']

    def completion_tensors(self, prompts: list, max_new_tokens: int, 
                          temperature: float, top_p: float):
        """Generate completions for multiple prompts."""
        self.model.eval()
        formatted_prompts = []
        
        # Format prompts with chat template if needed
        if self.use_chat_template:
            for prompt in prompts:
                messages = self._create_messages(prompt, self.available_tool_calls)

                # Apply chat template with tools if available
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
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return output_tensors, inputs['input_ids']

    def _create_messages(self, prompt: str, available_call: int) -> List[Dict[str, str]]:
        """Create messages for chat template with tool call awareness."""
        
        if available_call <= 0:
            if self.saved_context:
                user_message = f"{self.saved_context}\n\n{prompt}\n\nBased on the above API document snippets, examples and the signature, generate the missing implementation:"
            else:
                user_message = f"{prompt}\n\nBased on the above examples and the signature, generate the missing implementation:"
        else:
            if self.saved_context:
                user_message = f"{self.saved_context}\n\n{prompt}\n\nBased on the above API document snippets, examples and the signature, what is the most needed additional information to make the implementation complete?"
            else:
                user_message = f"{prompt}\n\nBased on the above examples and the signature, what is the most needed additional information to make the implementation complete?"

        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
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
        self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, stop
    ):   
        prompts = [prompt.strip() for prompt in prompts]
        
        # Use multi-generation RAG if enabled, otherwise fallback to original logic
        if self.use_rag and hasattr(self, 'multi_gen_engine'):
            return self._multi_generation_completions(prompts, max_tokens, temperature, top_p, stop)
        else:
            return self._original_completions(prompts, max_tokens, temperature, top_p, stop)
    
    def _multi_generation_completions(self, prompts: List[str], max_tokens: int, 
                                    temperature: float, top_p: float, stop) -> List[Dict[str, Any]]:
        """Multi-generation RAG completion."""
        results = []
        
        # Process each prompt with multi-generation engine
        for prompt in prompts:
            try:
                completion_result = self.multi_gen_engine.generate_with_tools(
                    prompt, max_tokens, temperature, top_p
                )
                
                # Format result for compatibility
                result = {
                    "pre_completion": "\n".join(completion_result.pre_completion),
                    "post_completion": completion_result.post_completion or "",
                    "tool_calls": completion_result.tool_calls_made,
                    "context": completion_result.context_accumulated,
                    "success": completion_result.success
                }
                
                results.append(result)
                    
            except Exception as e:
                # Handle errors gracefully
                result = {
                    "pre_completion": f"Error during generation: {str(e)}",
                    "post_completion": "",
                    "tool_calls": [],
                    "context": "",
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
                    "pre_completion": pre_completion,
                    "post_completion": post_completion,
                    "tool_calls": [],
                    "context": self.saved_context,
                    "success": True
                }
                results.append(result)
                
            except Exception as e:
                result = {
                    "pre_completion": f"Error: {str(e)}",
                    "post_completion": "",
                    "tool_calls": [],
                    "context": "",
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
    args.add_argument("--doc-path", type=str,
                     help="Path to the PDF document for RAG")
    args.add_argument("--embedding-model", type=str, 
                     default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Embedding model for RAG")
    args.add_argument("--k", type=int, default=3,
                     help="Number of documents to retrieve")
    args.add_argument("--max-tool-calls", type=int, default=3,
                     help="Maximum number of tool calls allowed")
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
    if args.use_rag and args.doc_path:
        rag_tools = RAGTools(
            doc_path=args.doc_path,
            embedding_model=args.embedding_model,
            k=args.k
        )
        print(f"RAG tools initialized with {len(rag_tools.get_tools())} tools")

    model = Model(
        args.name, 
        args.revision,
        model_kwargs=model_kwargs,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        use_chat_template=args.use_chat_template,
        use_rag=args.use_rag,
        rag_tools=rag_tools,
        max_tool_calls=args.max_tool_calls
    )

    name = do_name_override(args)
    make_main(args, name, model.completions)


if __name__ == "__main__":
    main()