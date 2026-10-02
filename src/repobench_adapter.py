from typing import Any, Dict, List

from .ast_parser import ASTParser, CodeChunk
from .repobench_loader import RepoBenchLoader


class RepoBenchAdapter:
    """
    Converts RepoBench's structured cross-file context
    into the CodeChunk representation used by the existing
    AST / Graph / Retrieval pipeline.
    """

    def __init__(self, parser: ASTParser | None = None):
        self.parser = parser or ASTParser()

    def build_retrieval_chunks(
        self,
        sample: Dict[str, Any]
    ) -> List[CodeChunk]:
        """
        Convert one RepoBench sample's context into CodeChunks.
        """

        context_items = RepoBenchLoader.get_context(sample)

        if not context_items:
            return []

        files_data = []

        repo_name = RepoBenchLoader.get_repo_name(sample)

        for index, item in enumerate(context_items):

            # RepoBench context normally contains structured
            # information such as path/name/code.
            if isinstance(item, dict):
                file_path = (
                    item.get("path")
                    or item.get("file_path")
                    or item.get("filepath")
                    or f"context_{index}.py"
                )

                code = (
                    item.get("code")
                    or item.get("content")
                    or item.get("snippet")
                    or ""
                )

            else:
                file_path = f"context_{index}.py"
                code = str(item)

            if not code.strip():
                continue

            # Add repo name to avoid collisions between repositories.
            unique_path = f"{repo_name}/{file_path}"

            files_data.append({
                "file_path": unique_path,
                "code": code,
            })

        if not files_data:
            return []

        chunks = self.parser.parse_repository(files_data)

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