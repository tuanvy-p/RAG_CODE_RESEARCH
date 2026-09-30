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
        
        selected_model = (
            model_name
            or os.getenv("LLM_MODEL")
            or getattr(config.model, "llm_model", "Qwen/Qwen2.5-Coder-7B-Instruct")
        )
        
        self.model_name = str(selected_model).strip()
        self.temperature = temperature if temperature is not None else getattr(config.model, "temperature", 0.2)
        self.device = device or getattr(config.model, 'device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

        print(f"[*] Loading Local LLM Model: `{self.model_name}` on `{self.device}`...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        device_map_target = {"": self.device} if "cuda" in self.device else "auto"

        model_kwargs = {
            "device_map": device_map_target,
            "trust_remote_code": True,
        }

        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            print("[CodeGenerator] Using bitsandbytes 4-bit quantization.")
        except Exception as e:
            print(f"[Warning] 4-bit quantization unavailable ({e}). Falling back to torch.float16.")
            model_kwargs["torch_dtype"] = torch.float16 if "cuda" in self.device else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        print(f"[SUCCESS] Model `{self.model_name}` loaded successfully on `{self.device}`!")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_line_level: bool = False
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        
        if max_tokens is not None:
            tokens = max_tokens
        elif is_line_level:
            tokens = 64
        else:
            tokens = getattr(config.model, 'max_tokens', 512)

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
            # 1. Encode prompt với truncation bảo vệ VRAM/OOM
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3500
            ).to(self.model.device)

            input_ids = inputs["input_ids"]

            gen_kwargs = {
                "max_new_tokens": tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if temp > 0:
                gen_kwargs["temperature"] = temp
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            # 2. Gọi trực tiếp model.generate (chỉ dùng max_new_tokens, KHÔNG dùng max_length)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )

            # 3. Trích xuất CHÍNH XÁC các token mới sinh (loại bỏ prompt tokens)
            generated_tokens = outputs[0][input_ids.shape[1]:]
            raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

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
        is_line_level: bool = False,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        expand_graph: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Supports Ablation Study parameters: dense_weight, sparse_weight, and expand_graph.
        """
        history: List[Dict[str, Any]] = []
        current_prediction = ""

        # Lấy tham số cấu hình mặc định nếu không truyền từ bên ngoài
        d_weight = dense_weight if dense_weight is not None else getattr(config.retriever, "dense_weight", 0.5)
        s_weight = sparse_weight if sparse_weight is not None else getattr(config.retriever, "sparse_weight", 0.5)
        exp_graph = expand_graph if expand_graph is not None else True

        lines = prefix_code.splitlines()
        window_size = getattr(config.repocoder, "sliding_window_size", 20)
        current_query = "\n".join(lines[-window_size:]) if len(lines) > window_size else prefix_code

        for iteration in range(max_iterations):
            # 1. Retrieve Context với tham số Ablation
            retrieved_chunks = retriever.retrieve(
                query=current_query,
                top_k=top_k,
                dense_weight=d_weight,
                sparse_weight=s_weight,
                expand_graph=exp_graph,
                hops=getattr(config.retriever, "graph_expansion_hops", 1)
            )

            # 2. Build prompt
            prompt = PromptBuilder.build_completion_prompt(
                prefix_code=prefix_code,
                context_chunks=retrieved_chunks,
                file_path=file_path
            )

            # 3. Generate completion
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
            
            pred_lines = current_prediction.splitlines()[:5]
            current_query = f"{current_query}\n" + "\n".join(pred_lines)

        return {
            "final_code": current_prediction,
            "iterations_count": max_iterations,
            "history": history
        }

    @staticmethod
    def _clean_completion_output(raw_output: str, is_line_level: bool = False) -> str:
        text = raw_output.strip()
        
        if "<COMPLETION_START>" in text:
            text = text.split("<COMPLETION_START>")[-1].strip()

        if "```" in text:
            lines = text.splitlines()
            cleaned_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    continue
                cleaned_lines.append(line)
            text = "\n".join(cleaned_lines).strip()

        if is_line_level:
            lines = [l for l in text.splitlines() if l.strip()]
            return lines[0].strip() if lines else ""

        return text

    def test_connection(self) -> bool:
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