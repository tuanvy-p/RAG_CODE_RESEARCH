from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
import re
import ast

def _initialize_tree_sitter():
    """
    Safely initializes Tree-sitter Language and Parser across different versions
    (tree-sitter-languages, tree-sitter >= 0.22, tree-sitter <= 0.21).
    """
    # Strategy 1: tree_sitter_languages (Most cross-platform compatible)
    try:
        from tree_sitter_languages import get_language, get_parser
        lang = get_language("python")
        parser = get_parser("python")
        return lang, parser, True
    except Exception:
        pass

    # Strategy 2: tree_sitter_python
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        raw_lang = tspython.language()

        lang = None
        if hasattr(raw_lang, "__class__") and raw_lang.__class__.__name__ == "Language":
            lang = raw_lang
        else:
            try:
                lang = Language(raw_lang)
            except Exception:
                lang = raw_lang

        parser = None
        try:
            parser = Parser(lang)
        except Exception:
            try:
                parser = Parser()
                parser.set_language(lang)
            except Exception:
                pass

        if parser is not None:
            return lang, parser, True
    except Exception:
        pass

    return None, None, False


PY_LANGUAGE, _GLOBAL_PARSER, _TREE_SITTER_AVAILABLE = _initialize_tree_sitter()

try:
    from tree_sitter import Node
except Exception:
    Node = Any


@dataclass
class CodeChunk:
    """
    Represents a syntax-aware semantic chunk extracted from an AST.
    """
    chunk_id: str                      # Unique ID: e.g. "src/utils.py::format_data"
    file_path: str                     # Path to source file
    name: str                          # Name of function, class, or module block
    node_type: str                     # 'function', 'method', 'class', 'import_block', 'standalone'
    code: str                          # Original raw source code of the chunk
    start_line: int                    # 1-indexed start line
    end_line: int                      # 1-indexed end line
    signature: Optional[str] = None    # Function/class signature header
    docstring: Optional[str] = None    # Extracted docstring if available
    parent_class: Optional[str] = None # Enclosing class name (if this is a method)
    calls: Set[str] = field(default_factory=set)       # Set of called functions/methods
    imports: Set[str] = field(default_factory=set)     # Imported modules or symbols

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "name": self.name,
            "node_type": self.node_type,
            "code": self.code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent_class": self.parent_class,
            "calls": list(self.calls),
            "imports": list(self.imports),
        }

    def get_embedding_text(self) -> str:
        """
        Formats the chunk into an information-rich representation for Dense Vector Embedding.
        """
        header = f"# File: {self.file_path}\n# Type: {self.node_type} | Name: {self.name}"
        if self.parent_class:
            header += f" | Class: {self.parent_class}"
        if self.signature:
            header += f"\n# Signature: {self.signature}"
        if self.docstring:
            header += f"\n# Docstring: {self.docstring}"
        return f"{header}\n{self.code}"


class ASTParser:
    """
    AST Parser for Python source code supporting both Tree-sitter and Python standard AST.
    Performs semantic code slicing (classes, functions, methods) and extracts
    call graphs and import dependencies.
    """

    def __init__(self):
        self.parser = _GLOBAL_PARSER

    def parse_file(self, file_path: str | Path) -> List[CodeChunk]:
        """Reads and parses a source file into semantic chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        return self.parse_code(code, file_path=str(path.as_posix()))

    def parse_code(self, code: str, file_path: str = "snippet.py") -> List[CodeChunk]:
        """Parses Python source code string into semantic chunks."""
        if not code.strip():
            return []

        # Strategy 1: Tree-sitter
        if self.parser is not None:
            try:
                tree = self.parser.parse(bytes(code, "utf-8"))
                if tree and tree.root_node:
                    root_node = tree.root_node
                    lines = code.splitlines()

                    chunks: List[CodeChunk] = []
                    global_imports: Set[str] = set()

                    # Step 1: Collect top-level imports
                    for child in root_node.children:
                        if child.type in ("import_statement", "import_from_statement"):
                            global_imports.update(self._parse_import_statement(child, code))

                    if global_imports:
                        import_chunk = self._create_import_chunk(root_node, code, file_path, global_imports)
                        if import_chunk:
                            chunks.append(import_chunk)

                    # Step 2: Extract functions, classes
                    for child in root_node.children:
                        self._process_node(child, code, lines, file_path, chunks, parent_class=None)

                    if chunks:
                        return chunks
            except Exception:
                pass

        # Strategy 2: Python Standard AST (Zero failure rate)
        return self._parse_code_with_python_ast(code, file_path)

    # =========================================================================
    # Python Standard Library AST Parser (Rock-solid Fallback)
    # =========================================================================

    def _parse_code_with_python_ast(self, code: str, file_path: str) -> List[CodeChunk]:
        """Parses code using Python's built-in `ast` module."""
        try:
            tree = ast.parse(code)
        except Exception:
            return []

        lines = code.splitlines()
        chunks: List[CodeChunk] = []
        global_imports: Set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    global_imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    global_imports.add(f"{mod}.{alias.name}" if mod else alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._create_ast_func_chunk(node, lines, code, file_path, parent_class=None))
            elif isinstance(node, ast.ClassDef):
                chunks.append(self._create_ast_class_chunk(node, lines, code, file_path))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(self._create_ast_func_chunk(item, lines, code, file_path, parent_class=node.name))

        if global_imports:
            import_lines = [l for l in lines if l.strip().startswith(("import ", "from "))]
            chunks.append(CodeChunk(
                chunk_id=f"{file_path}::__imports__",
                file_path=file_path,
                name="__imports__",
                node_type="import_block",
                code="\n".join(import_lines) if import_lines else "import ...",
                start_line=1,
                end_line=min(15, len(lines)),
                imports=global_imports
            ))

        return chunks

    def _create_ast_func_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: List[str],
        code: str,
        file_path: str,
        parent_class: Optional[str]
    ) -> CodeChunk:
        func_name = node.name
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line + 5)
        chunk_code = self._get_lines_slice(lines, start_line, end_line)

        # Signature
        first_line = lines[start_line - 1].strip() if 0 <= start_line - 1 < len(lines) else f"def {func_name}():"
        docstring = ast.get_docstring(node)
        calls = self._extract_ast_calls(node)

        chunk_id = f"{file_path}::{parent_class}.{func_name}" if parent_class else f"{file_path}::{func_name}"
        node_type = "method" if parent_class else "function"

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            name=func_name,
            node_type=node_type,
            code=chunk_code,
            start_line=start_line,
            end_line=end_line,
            signature=first_line,
            docstring=docstring,
            parent_class=parent_class,
            calls=calls
        )

    def _create_ast_class_chunk(
        self,
        node: ast.ClassDef,
        lines: List[str],
        code: str,
        file_path: str
    ) -> CodeChunk:
        class_name = node.name
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line + 10)
        chunk_code = self._get_lines_slice(lines, start_line, end_line)

        first_line = lines[start_line - 1].strip() if 0 <= start_line - 1 < len(lines) else f"class {class_name}:"
        docstring = ast.get_docstring(node)
        calls = self._extract_ast_calls(node)

        return CodeChunk(
            chunk_id=f"{file_path}::{class_name}",
            file_path=file_path,
            name=class_name,
            node_type="class",
            code=chunk_code,
            start_line=start_line,
            end_line=end_line,
            signature=first_line,
            docstring=docstring,
            parent_class=None,
            calls=calls
        )

    def _extract_ast_calls(self, node: ast.AST) -> Set[str]:
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        return calls

    # =========================================================================
    # Tree-sitter Extraction Methods
    # =========================================================================

    def _process_node(
        self,
        node: Node,
        code: str,
        lines: List[str],
        file_path: str,
        chunks: List[CodeChunk],
        parent_class: Optional[str] = None
    ):
        actual_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    actual_node = child
                    break

        if actual_node.type == "function_definition":
            chunk = self._create_function_chunk(
                node=node,
                func_node=actual_node,
                code=code,
                lines=lines,
                file_path=file_path,
                parent_class=parent_class
            )
            chunks.append(chunk)

        elif actual_node.type == "class_definition":
            class_name = self._get_node_identifier(actual_node, "name", code) or "AnonymousClass"
            class_chunk = self._create_class_chunk(
                node=node,
                class_node=actual_node,
                code=code,
                lines=lines,
                file_path=file_path,
                class_name=class_name
            )
            chunks.append(class_chunk)

            body_node = actual_node.child_by_field_name("body")
            if body_node:
                for child in body_node.children:
                    self._process_node(
                        node=child,
                        code=code,
                        lines=lines,
                        file_path=file_path,
                        chunks=chunks,
                        parent_class=class_name
                    )

    def _create_function_chunk(
        self,
        node: Node,
        func_node: Node,
        code: str,
        lines: List[str],
        file_path: str,
        parent_class: Optional[str]
    ) -> CodeChunk:
        func_name = self._get_node_identifier(func_node, "name", code) or "anonymous_function"
        chunk_id = f"{file_path}::{parent_class}.{func_name}" if parent_class else f"{file_path}::{func_name}"
        node_type = "method" if parent_class else "function"

        start_line = node.start_point.row + 1
        end_line = node.end_point.row + 1
        chunk_code = self._get_lines_slice(lines, start_line, end_line)

        signature = self._extract_signature(func_node, code)
        docstring = self._extract_docstring(func_node, code)
        calls = self._extract_call_expressions(func_node, code)

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            name=func_name,
            node_type=node_type,
            code=chunk_code,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            parent_class=parent_class,
            calls=calls
        )

    def _create_class_chunk(
        self,
        node: Node,
        class_node: Node,
        code: str,
        lines: List[str],
        file_path: str,
        class_name: str
    ) -> CodeChunk:
        chunk_id = f"{file_path}::{class_name}"
        start_line = node.start_point.row + 1
        end_line = node.end_point.row + 1
        chunk_code = self._get_lines_slice(lines, start_line, end_line)

        superclasses_node = class_node.child_by_field_name("superclasses")
        bases = self._get_node_text(superclasses_node, code) if superclasses_node else ""
        signature = f"class {class_name}{bases}:"

        docstring = self._extract_docstring(class_node, code)
        calls = self._extract_call_expressions(class_node, code)

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            name=class_name,
            node_type="class",
            code=chunk_code,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            parent_class=None,
            calls=calls
        )

    def _create_import_chunk(
        self,
        root_node: Node,
        code: str,
        file_path: str,
        imports: Set[str]
    ) -> Optional[CodeChunk]:
        import_lines = []
        min_line = 1000000
        max_line = 0

        for child in root_node.children:
            if child.type in ("import_statement", "import_from_statement"):
                start_l = child.start_point.row + 1
                end_l = child.end_point.row + 1
                min_line = min(min_line, start_l)
                max_line = max(max_line, end_l)
                import_lines.append(self._get_node_text(child, code))

        if not import_lines:
            return None

        return CodeChunk(
            chunk_id=f"{file_path}::__imports__",
            file_path=file_path,
            name="__imports__",
            node_type="import_block",
            code="\n".join(import_lines),
            start_line=min_line,
            end_line=max_line,
            imports=imports
        )

    def _extract_signature(self, func_node: Node, code: str) -> str:
        body_node = func_node.child_by_field_name("body")
        if body_node:
            sig_text = code[func_node.start_byte : body_node.start_byte].strip()
            if sig_text.endswith(":"):
                return sig_text
            return f"{sig_text}:"
        name = self._get_node_identifier(func_node, "name", code) or "func"
        params = func_node.child_by_field_name("parameters")
        params_str = self._get_node_text(params, code) if params else "()"
        return f"def {name}{params_str}:"

    def _extract_docstring(self, node: Node, code: str) -> Optional[str]:
        body_node = node.child_by_field_name("body")
        if not body_node or not body_node.children:
            return None

        for child in body_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        doc_raw = self._get_node_text(sub, code)
                        return doc_raw.strip("\"' \n\t")
            elif child.type not in ("comment", "\n"):
                break
        return None

    def _extract_call_expressions(self, node: Node, code: str) -> Set[str]:
        calls = set()
        def _traverse(curr: Node):
            if curr.type == "call":
                func_node = curr.child_by_field_name("function")
                if func_node:
                    call_name = self._get_node_text(func_node, code).strip()
                    calls.add(call_name)
            for child in curr.children:
                _traverse(child)

        _traverse(node)
        return calls

    def _parse_import_statement(self, node: Node, code: str) -> Set[str]:
        imported_symbols = set()
        text = self._get_node_text(node, code)
        
        if text.startswith("import "):
            modules = text.replace("import ", "").split(",")
            for m in modules:
                mod_name = m.strip().split(" as ")[0].strip()
                if mod_name:
                    imported_symbols.add(mod_name)
        elif text.startswith("from "):
            match = re.match(r"from\s+([\w\.]+)\s+import\s+(.+)", text)
            if match:
                mod_base = match.group(1)
                symbols = match.group(2).split(",")
                for s in symbols:
                    sym_name = s.strip().split(" as ")[0].strip()
                    if sym_name and sym_name != "*":
                        imported_symbols.add(f"{mod_base}.{sym_name}")
                    else:
                        imported_symbols.add(mod_base)
        return imported_symbols

    def _get_node_identifier(self, node: Node, field_name: str, code: str) -> Optional[str]:
        target = node.child_by_field_name(field_name)
        return self._get_node_text(target, code) if target else None

    def _get_node_text(self, node: Optional[Node], code: str) -> str:
        if node is None:
            return ""
        return code[node.start_byte : node.end_byte]

    def _get_lines_slice(self, lines: List[str], start_line: int, end_line: int) -> str:
        s = max(0, start_line - 1)
        e = min(len(lines), end_line)
        return "\n".join(lines[s:e])
