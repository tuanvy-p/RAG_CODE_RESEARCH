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
    max_samples: int = None
):
    print("\n" + "=" * 70)
    print(f"🚀 RUNNING BENCHMARK: {dataset_name.upper()}")
    print(f"📁 Target Repo Path : {repo_dir}")
    print(f"📄 Benchmark File   : {benchmark_file}")
    print("=" * 70)

    # 1. Tự động sinh tên file output & zip checkpoint dựa theo dataset_name truyền vào
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    output_file = out_dir_path / f"eval_results_{dataset_name}.json"
    zip_name = f"backup_checkpoint_{dataset_name}"

    completed_samples = []
    processed_ids = set()

    # 2. Tự động load tiến trình cũ nếu file output tương ứng đã tồn tại
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                completed_samples = saved_data.get("samples", [])
                processed_ids = {s["sample_id"] for s in completed_samples if "sample_id" in s}
            print(f"[*] RESUMING CHECKPOINT: Found {len(completed_samples)} samples already evaluated.")
        except Exception as e:
            print(f"[Warning] Could not read existing checkpoint file: {e}")

    # 3. Index Repository
    files_data = RepoDataLoader.load_repo_files(repo_dir)
    parser = ASTParser()
    all_chunks = parser.parse_repository(files_data)

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(),
        bm25_retriever=BM25Retriever(),
        dependency_graph=DependencyGraph()
    )
    retriever.index_repository(all_chunks)

    # 4. Load Model Generator & Data Benchmark
    generator = CodeGenerator()
    benchmark_samples = RepoDataLoader.load_jsonl_benchmark(benchmark_file)

    if max_samples and max_samples > 0:
        benchmark_samples = benchmark_samples[:max_samples]

    print(f"[*] Total samples to evaluate: {len(benchmark_samples)}")

    evaluator = CodeEvaluator()

    # 5. Vòng lặp Đánh giá (Tự bỏ qua sample đã chạy)
    for idx, sample in enumerate(benchmark_samples, 1):
        sample_id = sample.get("metadata", {}).get("task_id", f"{dataset_name}_sample_{idx}")

        if sample_id in processed_ids:
            continue

        prefix_code = sample.get("prompt", "") or sample.get("prefix", "")
        ground_truth = sample.get("ground_truth", "") or sample.get("suffix", "")
        file_path = sample.get("metadata", {}).get("file_path", "target_file.py")

        result = generator.generate_with_repocoder_loop(
            retriever=retriever,
            prefix_code=prefix_code,
            file_path=file_path,
            max_iterations=config.repocoder.max_iterations,
            top_k=config.retriever.top_k,
            is_line_level=is_line_level
        )

        predicted_code = result["final_code"]
        metrics = evaluator.evaluate_single_sample(predicted_code, ground_truth)

        record = {
            "sample_id": sample_id,
            "file_path": file_path,
            "prefix_code": prefix_code,
            "ground_truth": ground_truth,
            "predicted_code": predicted_code,
            "metrics": metrics,
            "history": result.get("history", [])
        }

        completed_samples.append(record)
        processed_ids.add(sample_id)

        summary = evaluator.compute_aggregate_metrics(completed_samples)

        # Ghi đè cập nhật tiến trình vào đúng file JSON của dataset đó
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_name": dataset_name,
                "total_completed": len(completed_samples),
                "summary": summary,
                "samples": completed_samples
            }, f, indent=2, ensure_ascii=False)

        print(f"[{dataset_name}] Saved {len(completed_samples)}/{len(benchmark_samples)} | Current EM: {summary['exact_match']:.2f}% | CodeBLEU: {summary['mean_codebleu']:.2f}")

        # Đồng bộ Dataset Kaggle mỗi 10 mẫu
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
    
    # Cho phép truyền đường dẫn linh hoạt từ bên ngoài
    parser.add_argument("--name", type=str, required=True, help="Tên bộ dữ liệu (vd: function_2k, line_4k, my_custom_dataset)")
    parser.add_argument("--repo", type=str, required=True, help="Đường dẫn folder chứa codebase repository")
    parser.add_argument("--benchmark", type=str, required=True, help="Đường dẫn file .jsonl dữ liệu đánh giá")
    parser.add_argument("--is_line_level", action="store_true", help="Cờ đánh dấu nếu là bài test sinh 1 dòng code/API")
    parser.add_argument("--dataset_id", type=str, default="", help="Kaggle Dataset ID nếu muốn backup checkpoint lên mây")
    parser.add_argument("--samples", type=int, default=0, help="Số lượng mẫu tối đa muốn test (0 = chạy hết)")

    args = parser.parse_args()

    max_s = None if args.samples <= 0 else args.samples

    run_benchmark(
        dataset_name=args.name,
        repo_dir=args.repo,
        benchmark_file=args.benchmark,
        is_line_level=args.is_line_level,
        dataset_id=args.dataset_id,
        max_samples=max_s
    )