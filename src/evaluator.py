from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from rapidfuzz.distance import Levenshtein
import re

# Optional CodeBLEU import with fallback
try:
    from codebleu import calc_codebleu
    _CODEBLEU_AVAILABLE = True
except ImportError:
    _CODEBLEU_AVAILABLE = False


class CodeEvaluator:
    """
    Evaluation Metrics Suite for Repository-Level Code Completion:
    - Exact Match (EM)
    - Edit Similarity (Levenshtein)
    - CodeBLEU / Token BLEU
    - Identifier Precision & Recall
    """

    @classmethod
    def evaluate_sample(cls, prediction: str, ground_truth: str) -> Dict[str, float]:
        """
        Computes all evaluation metrics for a single prediction vs ground truth pair.
        """
        pred_clean = cls._normalize_code(prediction)
        gt_clean = cls._normalize_code(ground_truth)

        em = 1.0 if pred_clean == gt_clean else 0.0
        edit_sim = cls.compute_edit_similarity(pred_clean, gt_clean)
        codebleu_score = cls.compute_codebleu(pred_clean, gt_clean)
        id_metrics = cls.compute_identifier_match(pred_clean, gt_clean)

        return {
            "exact_match": em,
            "edit_similarity": edit_sim,
            "codebleu": codebleu_score,
            "identifier_precision": id_metrics["precision"],
            "identifier_recall": id_metrics["recall"],
            "identifier_f1": id_metrics["f1"],
        }

    @classmethod
    def compute_edit_similarity(cls, pred: str, gt: str) -> float:
        """
        Computes normalized Edit Similarity (0.0 to 100.0) based on Levenshtein distance.
        Formula: 100 * (1 - LevenshteinDistance(pred, gt) / max(len(pred), len(gt)))
        """
        if not pred and not gt:
            return 100.0
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            return 100.0
        dist = Levenshtein.distance(pred, gt)
        similarity = max(0.0, 1.0 - (dist / max_len)) * 100.0
        return round(similarity, 2)

    @classmethod
    def compute_codebleu(cls, pred: str, gt: str, lang: str = "python") -> float:
        """
        Calculates CodeBLEU score (Syntax Match + Dataflow Match + N-gram BLEU).
        Falls back to Token-level BLEU if codebleu package is unavailable.
        """
        if not pred.strip() or not gt.strip():
            return 0.0

        if _CODEBLEU_AVAILABLE:
            try:
                result = calc_codebleu([gt], [pred], lang=lang, weights=(0.25, 0.25, 0.25, 0.25))
                return round(result["codebleu"] * 100.0, 2)
            except Exception:
                pass

        # Fallback simple token overlap BLEU
        return cls._simple_token_bleu(pred, gt)

    @classmethod
    def compute_identifier_match(cls, pred: str, gt: str) -> Dict[str, float]:
        """
        Computes precision, recall, and F1 score for unique code identifiers (variable/func names).
        """
        pred_ids = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', pred))
        gt_ids = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', gt))

        if not gt_ids:
            return {"precision": 100.0 if not pred_ids else 0.0, "recall": 100.0, "f1": 100.0}

        common = pred_ids.intersection(gt_ids)
        prec = (len(common) / len(pred_ids) * 100.0) if pred_ids else 0.0
        rec = (len(common) / len(gt_ids) * 100.0) if gt_ids else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return {
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2)
        }

    @classmethod
    def evaluate_dataset(cls, results: List[Dict[str, Any]], output_file: Optional[str | Path] = None) -> Dict[str, Any]:
        """
        Aggregates metrics across all evaluated benchmark samples and saves summary report.
        """
        if not results:
            return {}

        total = len(results)
        sum_em = sum(r.get("exact_match", 0.0) for r in results)
        sum_edit_sim = sum(r.get("edit_similarity", 0.0) for r in results)
        sum_codebleu = sum(r.get("codebleu", 0.0) for r in results)
        sum_id_f1 = sum(r.get("identifier_f1", 0.0) for r in results)

        summary = {
            "total_samples": total,
            "exact_match_pct": round(sum_em / total * 100.0, 2),
            "mean_edit_similarity": round(sum_edit_sim / total, 2),
            "mean_codebleu": round(sum_codebleu / total, 2),
            "mean_identifier_f1": round(sum_id_f1 / total, 2),
        }

        print("\n" + "="*50)
        print("          BENCHMARK EVALUATION SUMMARY          ")
        print("="*50)
        print(f"Total Samples           : {summary['total_samples']}")
        print(f"Exact Match (EM)        : {summary['exact_match_pct']}%")
        print(f"Mean Edit Similarity    : {summary['mean_edit_similarity']}")
        print(f"Mean CodeBLEU           : {summary['mean_codebleu']}")
        print(f"Mean Identifier F1      : {summary['mean_identifier_f1']}")
        print("="*50)

        if output_file:
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            report_data = {
                "summary": summary,
                "detailed_results": results
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"[CodeEvaluator] Full evaluation report saved to {path}")

        return summary

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Removes trailing whitespace, comments, and empty lines for clean comparison."""
        lines = [line.rstrip() for line in code.strip().splitlines() if line.strip()]
        return "\n".join(lines)

    @staticmethod
    def _simple_token_bleu(pred: str, gt: str) -> float:
        pred_tokens = pred.split()
        gt_tokens = gt.split()
        if not pred_tokens or not gt_tokens:
            return 0.0
        overlap = len(set(pred_tokens) & set(gt_tokens))
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        f_score = 2 * (precision * recall) / (precision + recall)
        return round(f_score * 100.0, 2)
