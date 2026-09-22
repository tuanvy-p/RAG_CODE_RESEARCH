from typing import List, Dict, Optional, Any
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .ast_parser import CodeChunk
from .config import config

from dotenv import load_dotenv


class PromptBuilder:
    """
    Constructs clean, structured prompts for Code Completion and RAG.
    """

    SYSTEM_PROMPT = (
        "You are an expert repository-level Python code completion model.\n"
        "Your task is to generate the exact, syntactically correct code completion that should follow immediately after the provided code prefix.\n"
        "Guidelines:\n"
        "1. Utilize the provided repository context (functions, classes, dependencies) to accurately use existing APIs, types, and variables.\n"
        "2. Return ONLY the raw code continuation without conversational explanations, commentary, or surrounding markdown fences unless necessary."
    )

    @classmethod
    def build_context_block(cls, chunks: List[CodeChunk]) -> str:
        if not chunks:
            return ""

        context_parts = ["# --- RELEVANT REPOSITORY CONTEXT ---"]
        for idx, chunk in enumerate(chunks, 1):
            header = f"# Context {idx} | File: {chunk.file_path} | Type: {chunk.node_type} ({chunk.name})"
            if chunk.signature:
                header += f"\n# Signature: {chunk.signature}"
            
            # Cắt bớt code nếu chunk quá dài (> 30 dòng)
            code_lines = chunk.code.splitlines()
            if len(code_lines) > 30:
                truncated_code = "\n".join(code_lines[:30]) + "\n# ... (truncated)"
            else:
                truncated_code = chunk.code

            context_parts.append(f"{header}\n{truncated_code}\n")
    
        context_parts.append("# --- END OF REPOSITORY CONTEXT ---\n")
        return "\n".join(context_parts)

    @classmethod
    def build_completion_prompt(
        cls,
        prefix_code: str,
        context_chunks: Optional[List[CodeChunk]] = None,
        file_path: str = "current_file.py"
    ) -> str:
        context_block = cls.build_context_block(context_chunks or [])
        
        prompt = (
            f"{context_block}\n"
            f"# Target File: {file_path}\n"
            f"# Please complete the code immediately following this prefix:\n"
            f"```python\n"
            f"{prefix_code}\n"
            f"<COMPLETION_START>"
        )
        return prompt


class CodeGenerator:
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None
    ):
        load_dotenv()
        
        # Lấy tên model local từ config
        selected_model = (
            model_name
            or os.getenv("LLM_MODEL")
            or config.model.llm_model
            or "Qwen/Qwen2.5-Coder-7B-Instruct"
        )
        
        self.model_name = str(selected_model).strip()
        self.temperature = temperature if temperature is not None else config.model.temperature
        self.device = device or getattr(config.model, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[*] Loading Local LLM Model: `{self.model_name}` on `{self.device}`...")

        # Khởi tạo Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Cấu hình load model dạng FP16 hoặc INT4 để tối ưu VRAM
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        if getattr(config.model, 'use_4bit', False):
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        
        # Tạo pipeline sinh văn bản local
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        print(f"[SUCCESS] Model `{self.model_name}` loaded successfully!")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_line_level: bool = False
    ) -> str:
        """
        Runs local inference using Hugging Face pipeline.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else getattr(config.model, 'max_tokens', 512)

        # Định dạng prompt dạng Chat / System message tương thích với Qwen / Llama
        messages = [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            outputs = self.pipe(
                formatted_prompt,
                max_new_tokens=tokens,
                temperature=temp,
                do_sample=True if temp > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Lấy chuỗi mã được sinh ra (loại bỏ phần prompt ban đầu)
            full_generated = outputs[0]["generated_text"]
            raw_text = full_generated[len(formatted_prompt):]

            return self._clean_completion_output(raw_text, is_line_level=is_line_level)

        except Exception as e:
            print(f"\n[Error] Local generation error: {e}")
            return ""

    def generate_with_repocoder_loop(
        self,
        retriever: Any,
        prefix_code: str,
        file_path: str = "current_file.py",
        max_iterations: int = 1,
        top_k: int = 2,
        is_line_level: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the RepoCoder Iterative Retrieval-Generation Loop.
        """
        history: List[Dict[str, Any]] = []
        current_prediction = ""

        lines = prefix_code.splitlines()
        window_size = config.repocoder.sliding_window_size
        current_query = "\n".join(lines[-window_size:]) if len(lines) > window_size else prefix_code

        for iteration in range(max_iterations):
            # 1. Retrieve Context
            retrieved_chunks = retriever.retrieve(
                query=current_query,
                top_k=top_k,
                dense_weight=config.retriever.dense_weight,
                sparse_weight=config.retriever.sparse_weight,
                expand_graph=True,
                hops=config.retriever.graph_expansion_hops
            )

            # 2. Build prompt
            prompt = PromptBuilder.build_completion_prompt(
                prefix_code=prefix_code,
                context_chunks=retrieved_chunks,
                file_path=file_path
            )

            # 3. Generate completion locally
            current_prediction = self.generate(
                prompt=prompt,
                is_line_level=is_line_level
            )

            history.append({
                "iteration": iteration + 1,
                "query_used": current_query,
                "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
                "generated_code": current_prediction
            })
            
            current_query = f"{current_query}\n{current_prediction}"

        return {
            "final_code": current_prediction,
            "iterations_count": max_iterations,
            "history": history
        }

    @staticmethod
    def _clean_completion_output(raw_output: str, is_line_level: bool = False) -> str:
        """
        Strips markdown code fences, tags, and extraneous explanation headers.
        """
        text = raw_output.strip()
        
        if "<COMPLETION_START>" in text:
            text = text.split("<COMPLETION_START>")[-1].strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) > 0 and lines[0].startswith("```"):
                lines = lines[1:]
            if len(lines) > 0 and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if is_line_level:
            non_empty_lines = [l for l in text.splitlines() if l.strip()]
            return non_empty_lines[0] if non_empty_lines else ""

        return text

    def test_connection(self) -> bool:
        """
        Tests if local LLM model is ready in memory.
        """
        print(f"[*] Testing Local LLM inference with model: `{self.model_name}`...")
        try:
            res = self.generate(prompt="print('hello')", max_tokens=10)
            if res is not None:
                print(f"[SUCCESS] Local LLM `{self.model_name}` is ready!\n")
                return True
            return False
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed to run local model `{self.model_name}`!")
            print(f"Error details: {e}\n")
            return False