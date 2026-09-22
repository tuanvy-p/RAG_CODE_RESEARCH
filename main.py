import os
import argparse
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import config
from src.data_loader import RepoDataLoader
from src.ast_parser import ASTParser, CodeChunk
from src.dependency_graph import DependencyGraph
from src.retriever import HybridRetriever, DenseRetriever, BM25Retriever
from src.generator import CodeGenerator
from src.evaluator import CodeEvaluator
from src.checkpoint import backup_to_kaggle_dataset

# Nạp biến môi trường
load_dotenv(override=True)


def setup_vector_store_with_checkpoint() -> bool:
    """
    Kiểm tra và phục hồi Vector DB từ Kaggle Input Dataset nếu tồn tại.
    """
    checkpoint_input_path = Path("/kaggle/input/code-rag-checkpoint/chroma_db_backup.zip")
    
    if checkpoint_input_path.exists():
        print("[*] Found existing Kaggle Checkpoint! Unpacking Vector DB...")
        shutil.unpack_archive(checkpoint_input_path, config.chroma_db_dir)
        print("[SUCCESS] Vector DB restored successfully. Skipping re-indexing!")
        return True
    
    print("[*] No Checkpoint found. Will build Vector DB from scratch...")
    return False


def index_repository(repo_path: str | Path) -> Tuple[List[CodeChunk], HybridRetriever]:
    """
    Loads all Python files from target repo, parses AST chunks, and builds Hybrid Retriever.
    Includes automatic Kaggle Dataset backup.
    """
    print(f"\n==================================================")
    print(f"   INDEXING REPOSITORY: {repo_path}")
    print(f"==================================================")

    # 1. Load files
    files_data = RepoDataLoader.load_repo_files(repo_path)
    print(f"[*] Discovered {len(files_data)} Python files in repository.")

    # 2. Parse AST into Semantic Chunks
    parser = ASTParser()
    all_chunks: List[CodeChunk] = []
    
    for file_info in files_data:
        try:
            chunks = parser.parse_code(code=file_info["content"], file_path=file_info["file_path"])
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[Warning] Failed to parse {file_info['file_path']}: {e}")

    print(f"[*] Extracted {len(all_chunks)} semantic chunks (functions, classes, imports).")

    # 3. Build Hybrid Retriever (FAISS + BM25 + Dependency Graph)
    retriever = HybridRetriever()
    
    # Kiểm tra nếu đã có Checkpoint thì khôi phục, không thì index mới
    has_checkpoint = setup_vector_store_with_checkpoint()
    
    if not has_checkpoint:
        retriever.index_repository(all_chunks)
        
        # Tự động backup Vector DB lên Kaggle Dataset ngay sau khi vừa Index xong
        print("[*] Backing up fresh Vector DB to Kaggle Dataset...")
        backup_to_kaggle_dataset(
            target_dir=str(config.chroma_db_dir),
            dataset_id="tunvyphm/code-rag-checkpoint",  #  THAY THÀNH KAGGLE USERNAME CỦA BẠN
            zip_name="chroma_db_backup"
        )
    else:
        # Nếu đã khôi phục từ checkpoint, chỉ cần nạp lại graph và BM25 cho chunks
        retriever.index_repository(all_chunks)

    # Print Graph stats
    stats = retriever.graph.get_stats()
    print(f"[*] Dependency Graph Stats: {stats['total_chunks_indexed']} nodes, {stats['total_dependency_edges']} edges ({stats['edge_breakdown']}).")

    return all_chunks, retriever


def run_benchmark_eval(
    benchmark_jsonl: str | Path,
    target_repo: str | Path,
    max_samples: Optional[int] = None,
    iterations: int = 1,
    top_k: int = 2,
):
    """
    Runs end-to-end evaluation on benchmark dataset (e.g. RepoEval / CrossCodeEval).
    """
    # ------------------------------------------------------------------
    # BƯỚC 1: HEALTH CHECK LOCAL LLM TRƯỚC
    # ------------------------------------------------------------------
    print("\n[*] Initializing CodeGenerator & testing Local LLM connection...")
    generator = CodeGenerator()

    print("\n========== DEBUG MODEL ==========")
    print("LLM_MODEL environment :", repr(os.getenv("LLM_MODEL")))
    print("Config model           :", repr(config.model.llm_model))
    print("Generator model        :", repr(generator.model_name))
    print("Generator class file   :", generator.__class__.__module__)
    print("=================================\n")
    
    if hasattr(generator, "test_connection"):
        is_llm_ready = generator.test_connection()
    else:
        print(f"[*] Testing LLM locally with model `{generator.model_name}`...")
        test_res = generator.generate("print('test')", max_tokens=2)
        is_llm_ready = bool(test_res)

    if not is_llm_ready:
        print("\n[ABORT] Local LLM initialization failed! Stopping execution.")
        return

    # ------------------------------------------------------------------
    # BƯỚC 2: INDEX REPO HOẶC LOAD TỪ CHECKPOINT
    # ------------------------------------------------------------------
    all_chunks, retriever = index_repository(target_repo)

    # 3. Load benchmark samples
    print(f"\n[*] Loading benchmark dataset: {benchmark_jsonl}")
    samples = RepoDataLoader.load_jsonl_benchmark(benchmark_jsonl)
    
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    print(f"[*] Evaluating on {len(samples)} test cases...")

    results = []
    is_line_task = "line_level" in str(benchmark_jsonl).lower()

    for sample in tqdm(samples, desc="Evaluating Code Completion"):
        prompt_prefix = sample.get("prompt", "")
        ground_truth = sample.get("ground_truth", "") or sample.get("target_code", "")
        file_path = sample.get("file_path", "test_file.py")
        sample_id = sample.get("sample_id", "sample")

        # Run RepoCoder Iterative Loop
        gen_output = generator.generate_with_repocoder_loop(
            retriever=retriever,
            prefix_code=prompt_prefix,
            file_path=file_path,
            max_iterations=iterations,
            top_k=top_k,
            is_line_level=is_line_task
        )

        prediction = gen_output["final_code"]

        # Compute Metrics
        metrics = CodeEvaluator.evaluate_sample(prediction, ground_truth)
        metrics["sample_id"] = sample_id
        metrics["prediction"] = prediction
        metrics["ground_truth"] = ground_truth
        metrics["history"] = gen_output["history"]
        results.append(metrics)

    # 4. Summarize and export report
    output_report_path = config.outputs_dir / "evaluation_report.json"
    CodeEvaluator.evaluate_dataset(results, output_file=output_report_path)
    
    # Backup kết quả báo cáo lên Kaggle Dataset
    print("[*] Backing up evaluation results to Kaggle Dataset...")
    backup_to_kaggle_dataset(
        target_dir=str(config.outputs_dir),
        dataset_id="your_username/code-rag-results",  #  THAY THÀNH KAGGLE USERNAME CỦA BẠN
        zip_name="outputs_backup"
    )


def run_interactive_demo(target_repo: str | Path):
    """
    Interactive test mode allowing user to test code completion queries live.
    """
    generator = CodeGenerator()
    if hasattr(generator, "test_connection") and not generator.test_connection():
        print("[ABORT] LLM Connection failed.")
        return

    all_chunks, retriever = index_repository(target_repo)

    print("\n" + "="*50)
    print("   INTERACTIVE CODE COMPLETION (Type 'exit' to quit) ")
    print("="*50)

    while True:
        try:
            print("\nEnter prompt prefix code (End with an empty line or EOF):")
            lines = []
            while True:
                line = input()
                if not line and lines:
                    break
                if line.strip() == "exit":
                    return
                lines.append(line)

            prefix_code = "\n".join(lines)
            if not prefix_code.strip():
                continue

            print("\n[Thinking & Retrieving Context via AST Hybrid + RepoCoder Loop...]")
            output = generator.generate_with_repocoder_loop(
                retriever=retriever,
                prefix_code=prefix_code,
                max_iterations=1,
                top_k=2
            )

            print("\n" + "-"*40 + " RETRIEVED CONTEXTS " + "-"*40)
            for it in output["history"]:
                print(f"Iteration {it['iteration']}: Retrived chunks -> {it['retrieved_chunk_ids']}")

            print("\n" + "="*40 + " GENERATED CONTINUATION " + "="*40)
            print(output["final_code"])
            print("="*100)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


def main():
    # Model local mặc định nếu môi trường chưa thiết lập
    if "LLM_MODEL" not in os.environ:
        os.environ["LLM_MODEL"] = "Qwen/Qwen2.5-Coder-7B-Instruct"

    parser = argparse.ArgumentParser(description="RepoCoder AST: Tree-sitter + Dependency Graph + Hybrid RAG")
    parser.add_argument("--mode", choices=["demo", "benchmark"], default="demo", help="Execution mode")
    parser.add_argument("--repo", type=str, default=str(config.target_repos_dir), help="Path to target codebase")
    parser.add_argument("--benchmark", type=str, default="", help="Path to benchmark .jsonl dataset")
    parser.add_argument("--samples", type=int, default=None, help="Max test samples for benchmark evaluation")
    parser.add_argument("--iterations", type=int, default=1, help="RepoCoder iterative refinement loops")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k context chunks to retrieve")

    args = parser.parse_args()

    if args.mode == "benchmark":
        if not args.benchmark:
            print("[Error] Please specify --benchmark path/to/dataset.jsonl")
            return
        run_benchmark_eval(
            benchmark_jsonl=args.benchmark,
            target_repo=args.repo,
            max_samples=args.samples,
            iterations=args.iterations,
            top_k=args.top_k
        )
    else:
        run_interactive_demo(target_repo=args.repo)


if __name__ == "__main__":
    main()