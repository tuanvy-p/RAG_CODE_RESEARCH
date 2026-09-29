import os
import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import config
from src.data_loader import RepoDataLoader
from src.generator import CodeGenerator
from src.retriever import HybridRetriever, DenseRetriever, BM25Retriever
from src.dependency_graph import DependencyGraph
from src.ast_parser import ASTParser
from src.evaluator import CodeEvaluator
from src.checkpoint import backup_to_kaggle_dataset


def run_benchmark(
    dataset_name: str,
    repo_dir: str,
    benchmark_file: str,
    output_dir: str = "outputs",
    is_line_level: bool = False,
    dataset_id: str = "",
    max_samples: int = None,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    expand_graph: bool = True,
    top_k: int = 2
):
    print("\n" + "=" * 70)
    print(f"🚀 RUNNING BENCHMARK: {dataset_name.upper()}")
    print(f"📁 Target Repo Path : {repo_dir}")
    print(f"📄 Benchmark File   : {benchmark_file}")
    print(f"⚙️ Config -> Top-K: {top_k} | Dense: {dense_weight} | BM25: {sparse_weight} | AST Graph: {expand_graph}")
    print("=" * 70)

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    output_file = out_dir_path / f"eval_results_{dataset_name}.json"
    zip_name = f"backup_checkpoint_{dataset_name}"

    completed_samples = []
    processed_ids = set()

    # 1. Load tiến trình checkpoint cũ nếu có
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                completed_samples = saved_data.get("samples", [])
                # Chuẩn hóa cách lấy ID đã xử lý
                processed_ids = {
                    s.get("sample_id") or s.get("metadata", {}).get("task_id") 
                    for s in completed_samples 
                    if s.get("sample_id") or s.get("metadata", {}).get("task_id")
                }
            print(f"[*] RESUMING CHECKPOINT: Found {len(completed_samples)} samples already evaluated.")
        except Exception as e:
            print(f"[Warning] Could not read existing checkpoint file: {e}")

    # 2. Index Repository
    files_data = RepoDataLoader.load_repo_files(repo_dir)
    parser = ASTParser()
    all_chunks = parser.parse_repository(files_data)

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(),
        bm25_retriever=BM25Retriever(),
        dependency_graph=DependencyGraph()
    )
    retriever.index_repository(all_chunks)

    # 3. Load Model Generator & Data Benchmark
    generator = CodeGenerator()
    benchmark_samples = RepoDataLoader.load_benchmark_dataset(benchmark_file)

    if max_samples and max_samples > 0:
        benchmark_samples = benchmark_samples[:max_samples]

    print(f"[*] Total samples to evaluate: {len(benchmark_samples)}")

    evaluator = CodeEvaluator()

    # 4. Vòng lặp Đánh giá
    for idx, sample in enumerate(benchmark_samples, 1):
        # Định danh sample_id nhất quán
        sample_id = (
            sample.get("sample_id") 
            or sample.get("metadata", {}).get("task_id") 
            or sample.get("id") 
            or f"{dataset_name}_sample_{idx}"
        )

        if sample_id in processed_ids:
            continue

        # Lấy prompt và ground_truth với đa dạng keys
        prefix_code = (
            sample.get("prompt") 
            or sample.get("prefix") 
            or sample.get("context", "")
        )
        
        ground_truth = (
            sample.get("ground_truth") 
            or sample.get("suffix") 
            or sample.get("target") 
            or sample.get("code_snippet", "")
        )
        
        file_path = (
            sample.get("file_path") 
            or sample.get("metadata", {}).get("file_path", "target_file.py")
        )

        try:
            result = generator.generate_with_repocoder_loop(
                retriever=retriever,
                prefix_code=prefix_code,
                file_path=file_path,
                max_iterations=config.repocoder.max_iterations,
                top_k=top_k,
                is_line_level=is_line_level,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                expand_graph=expand_graph
            )

            predicted_code = result["final_code"]
            metrics = evaluator.evaluate_single_sample(predicted_code, ground_truth)
            history = result.get("history", [])

        except Exception as e:
            print(f"\n[WARNING] Error at sample {idx}/{len(benchmark_samples)} ({sample_id}): {e}")
            predicted_code = ""
            metrics = evaluator.evaluate_single_sample("", ground_truth)
            history = [{"error": str(e)}]

        record = {
            "sample_id": sample_id,
            "file_path": file_path,
            "prefix_code": prefix_code,
            "ground_truth": ground_truth,
            "predicted_code": predicted_code,
            "metrics": metrics,
            "history": history
        }

        completed_samples.append(record)
        processed_ids.add(sample_id)

        summary = evaluator.compute_aggregate_metrics(completed_samples)

        # Ghi đè tiến trình
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_name": dataset_name,
                "total_completed": len(completed_samples),
                "summary": summary,
                "samples": completed_samples
            }, f, indent=2, ensure_ascii=False)

        print(f"[{dataset_name}] Saved {len(completed_samples)}/{len(benchmark_samples)} | Current EM: {summary['exact_match']:.2f}% | CodeBLEU: {summary['mean_codebleu']:.2f}")

        # Đồng bộ Kaggle Dataset mỗi 10 mẫu
        if len(completed_samples) % 10 == 0 and dataset_id:
            try:
                backup_to_kaggle_dataset(
                    target_dir=str(out_dir_path),
                    dataset_id=dataset_id,
                    zip_name=zip_name
                )
            except Exception as e:
                print(f"[Warning] Sync error: {e}")

    print(f"\n[SUCCESS] COMPLETED BENCHMARK: {dataset_name.upper()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Code RAG Evaluation Engine")
    
    parser.add_argument("--name", type=str, required=True, help="Tên bộ dữ liệu (vd: function_2k, line_4k, repobench_c)")
    parser.add_argument("--repo", type=str, required=True, help="Đường dẫn folder chứa codebase repository")
    parser.add_argument("--benchmark", type=str, required=True, help="Đường dẫn file .jsonl dữ liệu đánh giá")
    parser.add_argument("--is_line_level", action="store_true", help="Cờ đánh dấu nếu là bài test sinh 1 dòng code/API")
    parser.add_argument("--dataset_id", type=str, default="", help="Kaggle Dataset ID nếu muốn backup checkpoint lên mây")
    parser.add_argument("--samples", type=int, default=0, help="Số lượng mẫu tối đa muốn test (0 = chạy hết)")

    parser.add_argument("--dense_weight", type=float, default=0.5, help="Trọng số Dense Retrieval (0.0 - 1.0)")
    parser.add_argument("--sparse_weight", type=float, default=0.5, help="Trọng số BM25 Retrieval (0.0 - 1.0)")
    parser.add_argument("--no_graph", action="store_true", help="Tắt tính năng AST Graph Expansion")
    parser.add_argument("--top_k", type=int, default=2, help="Số lượng code snippets trích xuất")

    args = parser.parse_args()

    max_s = None if args.samples <= 0 else args.samples

    run_benchmark(
        dataset_name=args.name,
        repo_dir=args.repo,
        benchmark_file=args.benchmark,
        is_line_level=args.is_line_level,
        dataset_id=args.dataset_id,
        max_samples=max_s,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        expand_graph=not args.no_graph,
        top_k=args.top_k
    )