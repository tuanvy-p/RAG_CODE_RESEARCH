from typing import Any, Dict, List

from .ast_parser import ASTParser, CodeChunk
from .repobench_loader import RepoBenchLoader
from pathlib import Path

class RepoBenchAdapter:
    """
    Converts RepoBench's structured cross-file context
    into the CodeChunk representation used by the existing
    AST / Graph / Retrieval pipeline.
    """

    def __init__(self, parser: ASTParser | None = None):
        self.parser = parser or ASTParser()

    def build_retrieval_chunks(self, sample, chunk_mode: str = "snippet"):
        repo = RepoBenchLoader.get_repo_name(sample)
        chunks = []
        for item in RepoBenchLoader.get_context(sample):
            code = item.get("snippet") or ""
            if not code.strip():
                continue
            i = item["idx"]
            path = f"{repo}/{item['path']}"

            parsed = self.parser.parse_code(code, file_path=path)   # có thể rỗng
            if chunk_mode == "ast":                                 # giữ làm dòng ablation
                chunks.extend(parsed)
                continue

            calls, imports = set(), set()
            for p in parsed:
                calls |= p.calls
                imports |= p.imports
            first = next((l.strip() for l in code.splitlines() if l.strip()), "")
            chunks.append(CodeChunk(
                chunk_id=f"{path}::ctx{i}", file_path=path,
                name=item.get("identifier") or Path(path).stem,
                node_type="snippet", code=code,
                start_line=1, end_line=len(code.splitlines()),
                signature=first, calls=calls, imports=imports,
            ))
        return chunks

    def build_dependency_graph_chunks(
        self,
        sample: Dict[str, Any]
    ) -> List[CodeChunk]:
        """
        Alias with a descriptive name for experiments
        involving AST + Dependency Graph.
        """
        return self.build_retrieval_chunks(sample)

    @staticmethod
    def build_retrieval_query(
        sample: Dict[str, Any]
    ) -> str:
        """
        Build a retrieval-only query.

        IMPORTANT:
        This query is NOT used as the model prefix.

        It combines:
            - cropped_code
            - target file path
            - import statements

        This is especially important for RepoBench because
        cropped_code can be only a docstring/header.
        """

        prefix = RepoBenchLoader.get_prefix(sample)
        file_path = RepoBenchLoader.get_file_path(sample)
        imports = RepoBenchLoader.get_import_statement(sample)

        parts = []

        if prefix:
            parts.append(prefix)

        if file_path:
            parts.append(f"Target file: {file_path}")

        if imports:
            parts.append(imports)

        return "\n".join(parts).strip()