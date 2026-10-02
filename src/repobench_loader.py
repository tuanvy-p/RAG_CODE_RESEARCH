from pathlib import Path
from typing import Any, Dict, List

from .data_loader import RepoDataLoader


class RepoBenchLoader:
    """
    Loader chuyên biệt cho RepoBench.

    RepoBench không cung cấp một source repository local để
    run_evaluation.py đọc bằng --repo.

    Thay vào đó, mỗi sample chứa:
        - repo_name
        - file_path
        - cropped_code
        - context
        - import_statement
        - next_line
    """

    @classmethod
    def load(cls, benchmark_file: str | Path) -> List[Dict[str, Any]]:
        """
        Load toàn bộ RepoBench dataset.

        Việc đọc Parquet và normalize record được ủy quyền
        cho RepoDataLoader hiện tại.
        """
        return RepoDataLoader.load_benchmark_dataset(benchmark_file)

    @classmethod
    def get_prefix(cls, sample: Dict[str, Any]) -> str:
        """
        Lấy code prefix thực sự được dùng làm input cho model.
        """
        return str(sample.get("cropped_code") or "").strip()

    @classmethod
    def get_ground_truth(cls, sample: Dict[str, Any]) -> str:
        """
        Lấy dòng code cần dự đoán.

        RepoBench line-level dùng next_line.
        """
        return str(sample.get("next_line") or sample.get("ground_truth") or "").strip()

    @classmethod
    def get_file_path(cls, sample: Dict[str, Any]) -> str:
        """
        Lấy đường dẫn file target.
        """
        return str(sample.get("file_path") or "target_file.py")

    @classmethod
    def get_repo_name(cls, sample: Dict[str, Any]) -> str:
        """
        Lấy tên repository.
        """
        return str(sample.get("repo_name") or "unknown_repo")

    @classmethod
    def get_import_statement(cls, sample: Dict[str, Any]) -> str:
        """
        Lấy import statement của sample.

        Import không phải target code.
        Nó được dùng để cải thiện retrieval query.
        """
        value = sample.get("import_statement")

        if value is None:
            return ""

        if isinstance(value, (list, tuple)):
            return "\n".join(str(x) for x in value)

        return str(value)

    @classmethod
    def get_context(cls, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Lấy cross-file context được RepoBench cung cấp.
        """
        context = sample.get("context")

        if context is None:
            return []

        return list(context)