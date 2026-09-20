from typing import Dict, List, Optional, Set, Any, Tuple
import networkx as nx
from pathlib import Path

from .ast_parser import CodeChunk


class DependencyGraph:
    """
    Repository-level Dependency & Call Graph.
    Constructs a directed graph of classes, functions, methods, and files,
    modeling CALLS, IMPORTS, CONTAINS, and INHERITS relationships.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        # Symbol Table: maps identifier name -> list of chunk_ids where it is defined
        # e.g., "format_data" -> ["src/utils.py::format_data"]
        # "User.save" -> ["src/models.py::User.save"]
        self.symbol_table: Dict[str, List[str]] = {}
        # File to chunks map: file_path -> list of chunk_ids
        self.file_to_chunks: Dict[str, List[str]] = {}

    def build_from_chunks(self, chunks: List[CodeChunk]):
        """
        Builds the entire repository dependency graph from a list of CodeChunk objects.
        """
        # Step 1: Register all nodes and populate the Symbol Table
        for chunk in chunks:
            self._register_chunk_node(chunk)

        # Step 2: Resolve and connect edges (CALLS, CONTAINS, IMPORTS, INHERITS)
        for chunk in chunks:
            self._connect_chunk_edges(chunk)

    def _register_chunk_node(self, chunk: CodeChunk):
        """Adds a CodeChunk as a node and indexes its symbols."""
        self.graph.add_node(
            chunk.chunk_id,
            chunk_id=chunk.chunk_id,
            name=chunk.name,
            file_path=chunk.file_path,
            node_type=chunk.node_type,
            code=chunk.code,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            signature=chunk.signature,
            docstring=chunk.docstring,
            parent_class=chunk.parent_class,
            calls=list(chunk.calls),
            imports=list(chunk.imports)
        )

        # Index in file_to_chunks
        self.file_to_chunks.setdefault(chunk.file_path, []).append(chunk.chunk_id)

        # Index in Symbol Table:
        # 1. Simple name: e.g. "save"
        self._index_symbol(chunk.name, chunk.chunk_id)

        # 2. Qualified name (if method): e.g. "User.save"
        if chunk.parent_class:
            qualified_name = f"{chunk.parent_class}.{chunk.name}"
            self._index_symbol(qualified_name, chunk.chunk_id)

        # 3. File-scoped symbol: e.g. "models.py::save"
        file_stem = Path(chunk.file_path).name
        self._index_symbol(f"{file_stem}::{chunk.name}", chunk.chunk_id)

    def _index_symbol(self, symbol_name: str, chunk_id: str):
        """Helper to append chunk_id to symbol list without duplicates."""
        if symbol_name not in self.symbol_table:
            self.symbol_table[symbol_name] = []
        if chunk_id not in self.symbol_table[symbol_name]:
            self.symbol_table[symbol_name].append(chunk_id)

    def _connect_chunk_edges(self, chunk: CodeChunk):
        """Analyzes calls, parent classes, and imports to add directed edges."""
        source_id = chunk.chunk_id

        # 1. CONTAINS relationship (Class -> Method)
        if chunk.parent_class:
            parent_class_id = f"{chunk.file_path}::{chunk.parent_class}"
            if self.graph.has_node(parent_class_id):
                self.graph.add_edge(parent_class_id, source_id, relation="CONTAINS")

        # 2. CALLS relationships
        for call_expr in chunk.calls:
            resolved_target_ids = self._resolve_call_target(chunk, call_expr)
            for target_id in resolved_target_ids:
                if target_id != source_id:
                    self.graph.add_edge(source_id, target_id, relation="CALLS", call_name=call_expr)

        # 3. IMPORTS relationships
        for import_symbol in chunk.imports:
            resolved_import_ids = self.symbol_table.get(import_symbol, [])
            for target_id in resolved_import_ids:
                if target_id != source_id:
                    self.graph.add_edge(source_id, target_id, relation="IMPORTS")

    def _resolve_call_target(self, caller_chunk: CodeChunk, call_expr: str) -> List[str]:
        """
        Resolves a call expression (e.g. 'self.helper', 'utils.format', 'compute_loss')
        to candidate target chunk IDs in the repository.
        """
        targets: List[str] = []

        # Case A: self.method_name or cls.method_name inside a class method
        if (call_expr.startswith("self.") or call_expr.startswith("cls.")) and caller_chunk.parent_class:
            method_name = call_expr.split(".", 1)[1]
            # Try same class method first: Class.method_name
            scoped_name = f"{caller_chunk.parent_class}.{method_name}"
            if scoped_name in self.symbol_table:
                return self.symbol_table[scoped_name]
            # Fallback to simple method name
            if method_name in self.symbol_table:
                return self.symbol_table[method_name]

        # Case B: module.function_name or object.method_name (e.g. utils.format_data)
        if "." in call_expr:
            # Try exact match: "module.func" or "Class.method"
            if call_expr in self.symbol_table:
                return self.symbol_table[call_expr]
            
            # Try last component: "format_data"
            base_func = call_expr.split(".")[-1]
            if base_func in self.symbol_table:
                # Prefer matches within the same file or imported files
                candidate_ids = self.symbol_table[base_func]
                same_file_matches = [cid for cid in candidate_ids if cid.startswith(caller_chunk.file_path)]
                if same_file_matches:
                    return same_file_matches
                return candidate_ids

        # Case C: Direct function call (e.g. compute_loss)
        if call_expr in self.symbol_table:
            candidate_ids = self.symbol_table[call_expr]
            # Prioritize definition in the same file if exists
            same_file_matches = [cid for cid in candidate_ids if cid.startswith(caller_chunk.file_path)]
            if same_file_matches:
                return same_file_matches
            return candidate_ids

        return targets

    # =========================================================================
    # Context Expansion & Retrieval Utilities
    # =========================================================================

    def expand_context_for_seeds(
        self,
        seed_chunk_ids: List[str],
        hops: int = 1,
        max_expanded: int = 5,
        prefer_callees: bool = True
    ) -> List[str]:
        """
        Performs BFS graph expansion from seed chunk IDs to retrieve relevant context.
        - Successors (Callees): Function definitions called by the seed chunks.
        - Predecessors (Callers): Code that invokes the seed chunks (usage examples).
        """
        expanded_set: Set[str] = set()
        visited: Set[str] = set(seed_chunk_ids)
        current_layer: Set[str] = set(seed_chunk_ids)

        for _ in range(hops):
            next_layer: Set[str] = set()
            for node_id in current_layer:
                if not self.graph.has_node(node_id):
                    continue

                # 1. Successors (Callees - definitions needed by this code)
                successors = set(self.graph.successors(node_id)) - visited
                if prefer_callees:
                    for succ in successors:
                        if len(expanded_set) < max_expanded:
                            expanded_set.add(succ)
                            next_layer.add(succ)
                
                # 2. Predecessors (Callers - where this code is used)
                predecessors = set(self.graph.predecessors(node_id)) - visited
                for pred in predecessors:
                    if len(expanded_set) < max_expanded:
                        expanded_set.add(pred)
                        next_layer.add(pred)

                # 3. Class context (if method, include class definition chunk)
                parent_class = self.graph.nodes[node_id].get("parent_class")
                if parent_class:
                    file_path = self.graph.nodes[node_id].get("file_path", "")
                    class_chunk_id = f"{file_path}::{parent_class}"
                    if self.graph.has_node(class_chunk_id) and class_chunk_id not in visited:
                        expanded_set.add(class_chunk_id)

            visited.update(next_layer)
            current_layer = next_layer
            if len(expanded_set) >= max_expanded:
                break

        # Maintain deterministic order
        return list(expanded_set)[:max_expanded]

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the full node data for a chunk."""
        if self.graph.has_node(chunk_id):
            return dict(self.graph.nodes[chunk_id])
        return None

    def get_callees(self, chunk_id: str) -> List[str]:
        """Returns list of chunk_ids that this chunk directly calls."""
        if self.graph.has_node(chunk_id):
            return [target for _, target, d in self.graph.out_edges(chunk_id, data=True) if d.get("relation") == "CALLS"]
        return []

    def get_callers(self, chunk_id: str) -> List[str]:
        """Returns list of chunk_ids that call this chunk."""
        if self.graph.has_node(chunk_id):
            return [src for src, _, d in self.graph.in_edges(chunk_id, data=True) if d.get("relation") == "CALLS"]
        return []

    def get_file_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """Returns all chunk nodes belonging to a specific file."""
        chunk_ids = self.file_to_chunks.get(file_path, [])
        return [dict(self.graph.nodes[cid]) for cid in chunk_ids if self.graph.has_node(cid)]

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics of the constructed dependency graph."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        edge_relations: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "UNKNOWN")
            edge_relations[rel] = edge_relations.get(rel, 0) + 1

        return {
            "total_chunks_indexed": num_nodes,
            "total_dependency_edges": num_edges,
            "edge_breakdown": edge_relations,
            "unique_symbols_tracked": len(self.symbol_table),
            "files_covered": len(self.file_to_chunks)
        }
