from pathlib import Path
import argparse

from src.config import config
from src.repobench_loader import RepoBenchLoader
from src.repobench_adapter import RepoBenchAdapter
from src.retriever import (
    HybridRetriever,
    DenseRetriever,
    BM25Retriever
)
from src.dependency_graph import DependencyGraph
from src.generator import CodeGenerator
from src.evaluator import CodeEvaluator


def build_retriever(chunks):
    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(),
        bm25_retriever=BM25Retriever(),
        dependency_graph=DependencyGraph()
    )

    retriever.index_repository(chunks)

    return retriever


def run_repobench(
    benchmark_file: str,
    max_samples: int = 0,
    top_k: int = 2,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    expand_graph: bool = True
):

    print("\n" + "=" * 70)
    print("REPOBENCH EVALUATION")
    print("=" * 70)

    samples = RepoBenchLoader.load(benchmark_file)

    if max_samples > 0:
        samples = samples[:max_samples]

    print(f"Total samples: {len(samples)}")

    adapter = RepoBenchAdapter()
    generator = CodeGenerator(temperature=0.0)
    evaluator = CodeEvaluator()

    results = []

    for idx, sample in enumerate(samples, 1):

        print("\n" + "-" * 70)
        print(f"Sample {idx}/{len(samples)}")

        prefix_code = RepoBenchLoader.get_prefix(sample)
        ground_truth = RepoBenchLoader.get_ground_truth(sample)
        file_path = RepoBenchLoader.get_file_path(sample)

        retrieval_query = adapter.build_retrieval_query(sample)

        print(f"Repository : {RepoBenchLoader.get_repo_name(sample)}")
        print(f"Target     : {file_path}")
        print(f"Ground truth: {ground_truth}")

        # --------------------------------------------------
        # 1. Convert RepoBench context -> CodeChunk
        # --------------------------------------------------

        chunks = adapter.build_retrieval_chunks(sample)

        print(f"Context CodeChunks: {len(chunks)}")

        if not chunks:
            print("[WARNING] No retrieval chunks available.")
            continue

        # --------------------------------------------------
        # 2. Build RAG index for THIS sample
        # --------------------------------------------------

        retriever = build_retriever(chunks)

        # --------------------------------------------------
        # 3. Generate completion
        # --------------------------------------------------

        result = generator.generate_with_repocoder_loop(
            retriever=retriever,
            prefix_code=prefix_code,
            retrieval_query=retrieval_query,
            file_path=file_path,
            max_iterations=config.repocoder.max_iterations,
            top_k=top_k,
            is_line_level=True,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            expand_graph=expand_graph
        )

        predicted_code = result["final_code"]

        # --------------------------------------------------
        # 4. Evaluate
        # --------------------------------------------------

        metrics = evaluator.evaluate_single_sample(
            predicted_code,
            ground_truth
        )

        print(f"Prediction : {predicted_code}")
        print(f"EM         : {metrics}")

        results.append({
            "repo_name": RepoBenchLoader.get_repo_name(sample),
            "file_path": file_path,
            "prefix_code": prefix_code,
            "retrieval_query": retrieval_query,
            "ground_truth": ground_truth,
            "predicted_code": predicted_code,
            "metrics": metrics,
            "history": result.get("history", [])
        })

    print("\n" + "=" * 70)
    print("REPOBENCH FINISHED")
    print("=" * 70)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RepoBench evaluation pipeline"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=0
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=2
    )

    parser.add_argument(
        "--dense_weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--sparse_weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--no_graph",
        action="store_true"
    )

    args = parser.parse_args()

    run_repobench(
        benchmark_file=args.benchmark,
        max_samples=args.samples,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        expand_graph=not args.no_graph
    )