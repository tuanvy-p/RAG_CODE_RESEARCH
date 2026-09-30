import os
import json
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional


class RepoDataLoader:
    """
    Handles loading source code files from repositories and benchmark datasets (.jsonl).
    """

    SUPPORTED_EXTENSIONS = {".py"}
    IGNORED_DIRS = {"venv", ".venv", "__pycache__", ".git", ".pytest_cache", "node_modules", "dist", "build"}

    @classmethod
    def load_repo_files(cls, repo_dir: str | Path) -> List[Dict[str, str]]:
        """
        Recursively read all source code files in a target repository.
        Returns a list of dicts: [{"file_path": relative_or_absolute, "content": code_str}]
        """
        repo_path = Path(repo_dir)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository directory does not exist: {repo_path}")

        files_data = []
        for root, dirs, files in os.walk(repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in cls.IGNORED_DIRS and not d.startswith(".")]

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in cls.SUPPORTED_EXTENSIONS:
                    file_full_path = Path(root) / file
                    try:
                        with open(file_full_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        files_data.append({
                            "file_path": str(file_full_path.relative_to(repo_path).as_posix()),
                            "absolute_path": str(file_full_path.resolve()),
                            "content": content
                        })
                    except Exception as e:
                        print(f"[Warning] Failed to read {file_full_path}: {e}")

        return files_data

    @classmethod
    def load_benchmark_dataset(cls, file_path: str | Path) -> List[Dict[str, Any]]:
        """
        Loads a benchmark test dataset from a .json, .jsonl, or .parquet file (e.g., RepoEval, CrossCodeEval, RepoBench).
        Supports Parquet, JSON Array format, and JSONL (line-by-line) format.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        # 0. Hỗ trợ định dạng Parquet (.parquet)
        if path.suffix.lower() == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                records = df.to_dict(orient="records")
                return [cls._normalize_sample_record(item, idx) for idx, item in enumerate(records)]
            except ImportError:
                raise ImportError(
                    "Thư viện 'pandas' hoặc 'pyarrow' chưa được cài đặt. "
                    "Vui lòng chạy: pip install pandas pyarrow"
                )
            except Exception as e:
                raise RuntimeError(f"Lỗi khi đọc file Parquet {path}: {e}")

        # 1. Try parsing as a standard JSON file (list of dicts)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, list):
                return [cls._normalize_sample_record(item, idx) for idx, item in enumerate(content) if isinstance(item, dict)]
            elif isinstance(content, dict):
                return [cls._normalize_sample_record(content, 0)]
        except Exception:
            pass

        # 2. Fallback: Parse as JSON Lines (.jsonl)
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(cls._normalize_sample_record(data, line_idx))
                except json.JSONDecodeError as e:
                    print(f"[Warning] JSON decode error on line {line_idx}: {e}")

        return samples

    @classmethod
    def _normalize_sample_record(cls, data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """
        Normalizes RepoCoder, RepoEval, and RepoBench benchmark record formats.
        Flattens nested 'metadata' dictionary fields and maps dataset-specific keys.
        Safely handles strings, lists, and numpy arrays loaded from Parquet.
        """
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        def _to_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (list, tuple)):
                return "\n".join(str(x) for x in val if x is not None)
            # Hỗ trợ numpy array / pandas Series nếu có
            if hasattr(val, "tolist"):
                arr = val.tolist()
                if isinstance(arr, (list, tuple)):
                    return "\n".join(str(x) for x in arr if x is not None)
                return str(arr)
            return str(val)

        # 1. GROUND TRUTH (Hỗ trợ RepoCoder, RepoEval & RepoBench 'target'/'code_snippet')
        if "ground_truth" not in data or data["ground_truth"] is None:
            raw_gt = (
                data.get("target") 
                if data.get("target") is not None else
                data.get("code_snippet") 
                if data.get("code_snippet") is not None else
                metadata.get("ground_truth") 
                if metadata.get("ground_truth") is not None else
                data.get("target_code", "")
            )
            data["ground_truth"] = _to_str(raw_gt)
        else:
            data["ground_truth"] = _to_str(data["ground_truth"])

        # 2. PROMPT / PREFIX (Hỗ trợ RepoBench ghép 'context' + 'import_statement')
        if "prompt" not in data and "prefix" not in data:
            context_str = _to_str(data.get("context", "")).strip()
            imports_str = _to_str(data.get("import_statement", "")).strip()
            
            prompt_parts = []
            if context_str:
                prompt_parts.append(context_str)
            if imports_str:
                prompt_parts.append(imports_str)
            
            data["prompt"] = "\n".join(prompt_parts).strip()
        else:
            if "prompt" in data and data["prompt"] is not None:
                data["prompt"] = _to_str(data["prompt"])
            elif "prefix" in data and data["prefix"] is not None:
                data["prefix"] = _to_str(data["prefix"])

        # 3. FILE PATH & REPO NAME
        if "file_path" not in data or not data["file_path"]:
            fpath_tuple = metadata.get("fpath_tuple", [])
            if hasattr(fpath_tuple, "tolist"):
                fpath_tuple = fpath_tuple.tolist()

            if isinstance(fpath_tuple, (list, tuple)) and len(fpath_tuple) > 0:
                data["repo_name"] = str(fpath_tuple[0])
                data["file_path"] = "/".join(str(x) for x in fpath_tuple[1:]) if len(fpath_tuple) > 1 else str(fpath_tuple[0])
                data["full_fpath"] = "/".join(str(x) for x in fpath_tuple)
            else:
                data["file_path"] = str(data.get("fpath") or data.get("file_name") or f"sample_{idx}.py")
        else:
            data["file_path"] = str(data["file_path"])

        # 4. SAMPLE ID
        raw_id = data.get("id") or metadata.get("task_id") or data.get("sample_id") or f"sample_{idx}"
        data["sample_id"] = str(raw_id)
        
        return data

    # Alias for backward compatibility
    load_jsonl_benchmark = load_benchmark_dataset
