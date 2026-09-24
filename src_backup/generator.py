from typing import List, Dict, Optional, Any, Tuple
import re
import os
import time
from groq import Groq

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
            
            # Cắt bớt code nếu chunk quá dài (> 30 dòng) để tránh vượt quá ITPM limit
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
        """
        Combines repository context and current file prefix into the final prompt.
        """
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
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        # Nạp lại biến môi trường
        load_dotenv()
        
        # Đọc GROQ_API_KEY từ os.getenv hoặc config
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or getattr(config.model, 'api_key', None)
        
        # Thứ tự ưu tiên: Tham số truyền vào -> os.getenv("LLM_MODEL") -> config -> fallback default
        selected_model = (
            model_name
            or os.getenv("LLM_MODEL")
            or config.model.llm_model
            or "openai/gpt-oss-120b"
        )

            
        self.model_name = str(selected_model).strip()
        self.temperature = temperature if temperature is not None else config.model.temperature

        if not self.api_key:
            print("[Warning] GROQ_API_KEY is not set. Generator will fail until key is configured.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_line_level: bool = False,
        max_retries: int = 3
    ) -> str:
        """
        Calls Groq API with the given prompt, including auto-retry on Rate Limits.
        """
        if not self.client:
            raise ValueError("Groq API key is not configured. Please set GROQ_API_KEY in .env")

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else getattr(config.model, 'max_tokens', 512)
        stop_sequences = ["\n"] if is_line_level else None

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": PromptBuilder.SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temp,
                    max_completion_tokens=tokens,
                    stop=stop_sequences
                )
                
                raw_text = completion.choices[0].message.content or ""
                return self._clean_completion_output(raw_text, is_line_level=is_line_level)
                
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "rate_limit_exceeded" in err_msg:
                    print(f"\n[Warning] Rate limit hit (Attempt {attempt+1}/{max_retries}). Waiting 10s...")
                    time.sleep(10)
                else:
                    print(f"\n[Error] Groq API generation error: {e}")
                    break

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
        Executes the RepoCoder Iterative Retrieval-Generation Loop:
        - Round 0: Query = Prefix sliding window -> Retrieve Context -> Generate initial Code_0
        - Round 1..N: Query = Prefix + Code_{i-1} -> Retrieve updated Context -> Refine Code_i
        """
        history: List[Dict[str, Any]] = []
        current_prediction = ""

        # Extract initial query from prefix
        lines = prefix_code.splitlines()
        window_size = config.repocoder.sliding_window_size
        current_query = "\n".join(lines[-window_size:]) if len(lines) > window_size else prefix_code

        for iteration in range(max_iterations):
            # 1. Retrieve Context using current query
            retrieved_chunks = retriever.retrieve(
                query=current_query,
                top_k=top_k,
                dense_weight=config.retriever.dense_weight,
                sparse_weight=config.retriever.sparse_weight,
                expand_graph=True,
                hops=config.retriever.graph_expansion_hops
            )

            # 2. Build prompt with current retrieved context
            prompt = PromptBuilder.build_completion_prompt(
                prefix_code=prefix_code,
                context_chunks=retrieved_chunks,
                file_path=file_path
            )

            # 3. Generate completion via Groq
            current_prediction = self.generate(
                prompt=prompt,
                is_line_level=is_line_level
            )

            # Record iteration history
            history.append({
                "iteration": iteration + 1,
                "query_used": current_query,
                "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
                "generated_code": current_prediction
            })
            
            # 4. Formulate new query for next iteration
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
        
        # Remove <COMPLETION_START> if mirrored by LLM
        if "<COMPLETION_START>" in text:
            text = text.split("<COMPLETION_START>")[-1].strip()

        # Remove markdown code block wrapping (```python ... ``` or ``` ... ```)
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) > 0 and lines[0].startswith("```"):
                lines = lines[1:]
            if len(lines) > 0 and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Nếu là line level: Lấy duy nhất dòng code đầu tiên có nội dung
        if is_line_level:
            non_empty_lines = [l for l in text.splitlines() if l.strip()]
            return non_empty_lines[0] if non_empty_lines else ""

        return text

    def test_connection(self) -> bool:
        if not self.client:
            print("[Error] Groq API client chưa được khởi tạo. Kiểm tra GROQ_API_KEY trong .env")
            return False

        # In rõ model_name ra màn hình để kiểm tra
        print(f"[*] Testing LLM API connection with model: `{self.model_name}`...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, # Ép dùng đúng self.model_name
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=1
            )
            print(f"[SUCCESS] LLM connection OK! Model `{self.model_name}` sẵn sàng.\n")
            return True
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Kiểm tra LLM thất bại với model `{self.model_name}`!")
            print(f"Chi tiết lỗi: {e}\n")
            return False