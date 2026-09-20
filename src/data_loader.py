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
        Loads a benchmark test dataset from a .json or .jsonl file (e.g., RepoEval, CrossCodeEval).
        Supports both JSON Array format and JSONL (line-by-line) format.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

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
        Normalizes RepoCoder / RepoEval benchmark record formats.
        Flattens nested 'metadata' dictionary fields (ground_truth, fpath_tuple, etc.).
        """
        metadata = data.get("metadata", {})
        
        # Ground Truth
        if "ground_truth" not in data:
            data["ground_truth"] = metadata.get("ground_truth") or data.get("target_code", "")

        # File Path & Repo Name
        if "file_path" not in data:
            fpath_tuple = metadata.get("fpath_tuple", [])
            if fpath_tuple and isinstance(fpath_tuple, list):
                data["repo_name"] = fpath_tuple[0]
                data["file_path"] = "/".join(fpath_tuple[1:]) if len(fpath_tuple) > 1 else fpath_tuple[0]
                data["full_fpath"] = "/".join(fpath_tuple)
            else:
                data["file_path"] = data.get("fpath", f"sample_{idx}.py")

        data.setdefault("sample_id", metadata.get("task_id") or f"sample_{idx}")
        return data

    # Alias for backward compatibility
    load_jsonl_benchmark = load_benchmark_dataset
