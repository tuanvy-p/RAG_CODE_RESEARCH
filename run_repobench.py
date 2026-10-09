import argparse
import json
import random

from rapidfuzz import fuzz

from src.config import config
from src.repobench_loader import RepoBenchLoader
from src.repobench_adapter import RepoBenchAdapter
from src.retriever import HybridRetriever, DenseRetriever, BM25Retriever
from src.dependency_graph import DependencyGraph
from src.generator import CodeGenerator
from src.evaluator import CodeEvaluator


class NullRetriever:
    """Baseline không retrieval."""
    def index_repository(self, chunks):
        pass

    def retrieve(self, *args, **kwargs):
        return []


def build_retriever(chunks, dense):
    r = HybridRetriever(
        dense_retriever=dense,              # dùng lại model, không load lại mỗi sample
        bm25_retriever=BM25Retriever(),
        dependency_graph=DependencyGraph(),
    )
    r.index_repository(chunks)
    return r


def build_full_prefix(sample):
    imp = RepoBenchLoader.get_import_statement(sample)
    prefix = RepoBenchLoader.get_prefix(sample)
    return f"{imp}\n\n{prefix}" if imp.strip() else prefix


def find_gold_chunks(sample, chunks):
    gold = RepoBenchLoader.get_gold_index(sample)
    ctx = RepoBenchLoader.get_context(sample)
    # Cách 1: adapter đã đổi sang chunk_id dạng ...::ctx{i}
    found = [c for c in chunks if c.chunk_id.endswith(f"::ctx{gold}")]
    if found:
        return found
    # Cách 2 (adapter cũ): khớp theo đường dẫn file của snippet vàng
    repo = RepoBenchLoader.get_repo_name(sample)
    gp = f"{repo}/{ctx[gold]['path']}"
    return [c for c in chunks if c.file_path == gp]


def run_repobench(benchmark_file, max_samples=0, seed=42, mode="retrieval",
                  top_k=2, dense_weight=0.5, sparse_weight=0.5,
                  expand_graph=True, out_path="results.jsonl"):

    samples = RepoBenchLoader.load(benchmark_file)
    if max_samples > 0:
        samples = random.Random(seed).sample(samples, min(max_samples, len(samples)))
    print(f"Mode={mode} | samples={len(samples)} | seed={seed}")

    adapter = RepoBenchAdapter()
    generator = CodeGenerator(temperature=0.0)      # greedy
    dense = DenseRetriever()                        # tạo MỘT lần

    n, em_sum, es_sum, empty_gold = 0, 0, 0.0, 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples, 1):
            full_prefix = build_full_prefix(sample)
            gt = RepoBenchLoader.get_ground_truth(sample)
            file_path = RepoBenchLoader.get_file_path(sample)
            query = adapter.build_retrieval_query(sample)
            chunks = adapter.build_retrieval_chunks(sample)

            if mode == "none" or not chunks:
                retriever = NullRetriever()
            elif mode == "oracle":
                gold_chunks = find_gold_chunks(sample, chunks)
                if gold_chunks:
                    retriever = build_retriever(gold_chunks, dense)
                else:
                    empty_gold += 1
                    retriever = NullRetriever()
            else:
                retriever = build_retriever(chunks, dense)

            result = generator.generate_with_repocoder_loop(
                retriever=retriever,
                prefix_code=full_prefix,
                retrieval_query=query,
                file_path=file_path,
                max_iterations=config.repocoder.max_iterations,
                top_k=top_k,
                is_line_level=True,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                expand_graph=expand_graph,
            )

            pred = result["final_code"]
            pred_s, gt_s = pred.strip(), gt.strip()
            em = int(pred_s == gt_s)
            es = CodeEvaluator.compute_edit_similarity(pred_s, gt_s)    # công thức của project
            es_fuzz = fuzz.ratio(pred_s, gt_s)                          # cột phụ để so với paper

            n += 1
            em_sum += em
            es_sum += es

            fout.write(json.dumps({
                "sample_id": sample.get("sample_id"),
                "level": sample.get("level"),
                "mode": mode,
                "n_chunks": len(chunks),
                "gold_idx": RepoBenchLoader.get_gold_index(sample),
                "prompt_tail": full_prefix[-300:],
                "raw_output": result.get("raw_output"),
                "processed_output": pred,
                "ground_truth": gt,
                "em": em,
                "es": es,
                "es_fuzz": es_fuzz,
            }, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"[{idx}/{len(samples)}] EM={em} ES={es:.1f} | "
                  f"running EM={100 * em_sum / n:.2f} ES={es_sum / n:.2f}")

    print(f"\nFINAL mode={mode}: EM={100 * em_sum / n:.2f}  ES={es_sum / n:.2f}  (n={n})")
    if mode == "oracle":
        print(f"Số mẫu không tìm được chunk vàng: {empty_gold}/{n}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RepoBench evaluation pipeline")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["none", "oracle", "retrieval"], default="retrieval")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--dense_weight", type=float, default=0.5)
    p.add_argument("--sparse_weight", type=float, default=0.5)
    p.add_argument("--no_graph", action="store_true")
    p.add_argument("--out", default="results.jsonl")
    a = p.parse_args()

    run_repobench(a.benchmark, a.samples, a.seed, a.mode, a.top_k,
                  a.dense_weight, a.sparse_weight, not a.no_graph, a.out)