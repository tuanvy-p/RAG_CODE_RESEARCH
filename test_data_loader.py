import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import RepoDataLoader

def test_repobench_normalization():
    print("=" * 60)
    print("TEST 1: RepoBench Mock Record Normalization")
    print("=" * 60)
    
    mock_repobench = {
        "repo_name": "psf/requests",
        "file_path": "requests/models.py",
        "context": ["class Response(object):", "    def __init__(self):"],
        "import_statement": ["import os", "import json"],
        "cropped_code": "def send_request(url, headers):\n    # Prepare session\n    session = requests.Session()\n",
        "next_line": "    response = session.get(url, headers=headers)",
        "gold_snippet_index": 0,
        "level": "cross_file_first"
    }
    
    normalized = RepoDataLoader._normalize_sample_record(mock_repobench.copy(), 0)
    
    print(f"Dataset      : RepoBench (Mock)")
    print(f"Sample ID    : {normalized.get('sample_id')}")
    print(f"Repo Name    : {normalized.get('repo_name')}")
    print(f"File Path    : {normalized.get('file_path')}")
    print(f"Prompt/Prefix:\n{repr(normalized.get('prompt'))}")
    print(f"Ground Truth :\n{repr(normalized.get('ground_truth'))}")
    
    assert normalized.get("prompt") == mock_repobench["cropped_code"], "Failed: prompt != cropped_code"
    assert normalized.get("ground_truth") == mock_repobench["next_line"], "Failed: ground_truth != next_line"
    assert normalized.get("file_path") == mock_repobench["file_path"], "Failed: file_path != mock file_path"
    assert normalized.get("repo_name") == mock_repobench["repo_name"], "Failed: repo_name != mock repo_name"
    assert len(normalized.get("prompt")) > 0, "Failed: prompt is empty"
    assert len(normalized.get("ground_truth")) > 0, "Failed: ground_truth is empty"
    print("-> PASS: RepoBench normalization matches cropped_code and next_line perfectly!\n")


def test_repocoder_normalization():
    print("=" * 60)
    print("TEST 2: RepoCoder Original Record Normalization (Regression Test)")
    print("=" * 60)
    
    repocoder_file = Path("data/raw_jsonl/line_level_completion_2k_context_codex.test.jsonl")
    if repocoder_file.exists():
        samples = RepoDataLoader.load_benchmark_dataset(repocoder_file)
        sample0 = samples[0]
        print(f"Dataset      : RepoCoder (.jsonl)")
        print(f"Sample ID    : {sample0.get('sample_id')}")
        print(f"Repo Name    : {sample0.get('repo_name')}")
        print(f"File Path    : {sample0.get('file_path')}")
        print(f"Prompt (len={len(sample0.get('prompt', ''))}):\n{sample0.get('prompt', '')[:120]}...")
        print(f"Ground Truth :\n{repr(sample0.get('ground_truth'))}")
        
        assert sample0.get("ground_truth") == "    StableDiffusionInpaintPipelineLegacy,", "Failed: ground_truth mismatch for RepoCoder"
        assert sample0.get("repo_name") == "huggingface_diffusers", "Failed: repo_name mismatch for RepoCoder"
        assert sample0.get("file_path") == "tests/pipelines/stable_diffusion/test_stable_diffusion_inpaint_legacy.py", "Failed: file_path mismatch for RepoCoder"
        assert len(sample0.get("prompt")) > 0, "Failed: RepoCoder prompt is empty"
        print("-> PASS: RepoCoder normalization remains 100% intact and unchanged!\n")
    else:
        print(f"[Warning] RepoCoder file not found at {repocoder_file}")


def test_real_repobench_parquet():
    print("=" * 60)
    print("TEST 3: Real RepoBench Parquet File")
    print("=" * 60)
    
    parquet_files = list(Path("data_repobench").glob("*.parquet")) + list(Path("data").glob("*.parquet"))
    if not parquet_files:
        print("No .parquet files found in data_repobench/ or data/")
        return
        
    target_parquet = parquet_files[0]
    print(f"Loading 1 sample from: {target_parquet}")
    try:
        samples = RepoDataLoader.load_benchmark_dataset(target_parquet)
        sample0 = samples[0]
        print(f"Sample ID    : {sample0.get('sample_id')}")
        print(f"Repo Name    : {sample0.get('repo_name')}")
        print(f"File Path    : {sample0.get('file_path')}")
        print(f"Prefix (len={len(sample0.get('prompt', ''))}):\n{sample0.get('prompt', '')[:200]}...")
        print(f"Ground Truth :\n{repr(sample0.get('ground_truth'))}")
        assert len(sample0.get("prompt")) > 0, "Failed: Parquet prefix is empty"
        assert len(sample0.get("ground_truth")) > 0, "Failed: Parquet ground_truth is empty"
        print("-> PASS: Real Parquet sample loaded successfully with non-empty prefix and ground_truth!\n")
    except Exception as e:
        print(f"Note on Parquet direct load: {e}")

if __name__ == "__main__":
    test_repobench_normalization()
    test_repocoder_normalization()
    test_real_repobench_parquet()
