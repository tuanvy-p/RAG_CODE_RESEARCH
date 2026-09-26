import sys
import os

# Thêm thư mục hiện tại vào sys.path để tránh lỗi ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.ast_parser import CodeChunk
from src.retriever import HybridRetriever
from src.generator import CodeGenerator

def test_pipeline():
    print("=" * 50)
    print("🚀 BẮT ĐẦU TEST LOCAL PIPELINE")
    print("=" * 50)

    # 1. Kiểm tra Config
    print("\n[1/4] Kiểm tra Cấu hình:")
    print(f"  - LLM Model       : {config.model.llm_model}")
    print(f"  - Embedding Model : {config.model.embedding_model}")
    print(f"  - Device          : {config.model.device}")

    # 2. Tạo Mock Data
    print("\n[2/4] Khởi tạo dữ liệu mẫu (Mock Chunks)...")
    sample_chunks = [
        CodeChunk(
            chunk_id="file1_func1",
            file_path="src_test/utils.py",
            node_type="function",
            name="add_numbers",
            signature="def add_numbers(a, b):",
            code="def add_numbers(a, b):\n    return a + b",
            start_line=1,  # <--- BỔ SUNG
            end_line=2     # <--- BỔ SUNG
        ),
        CodeChunk(
            chunk_id="file1_func2",
            file_path="src_test/utils.py",
            node_type="function",
            name="multiply_numbers",
            signature="def multiply_numbers(a, b):",
            code="def multiply_numbers(a, b):\n    return a * b",
            start_line=4,  # <--- BỔ SUNG
            end_line=5     # <--- BỔ SUNG
        )
    ]
    print(f"  -> Đã tạo {len(sample_chunks)} chunks mẫu thành công.")

    # 3. Test Retriever
    print("\n[3/4] Test Retriever (BM25 + Dense Local Embedding)...")
    try:
        hybrid_retriever = HybridRetriever()
        hybrid_retriever.index_repository(sample_chunks)
        
        query = "function to calculate sum of two numbers"
        results = hybrid_retriever.retrieve(query=query, top_k=1)
        retrieved_name = results[0].name if results else "None"
        print(f"  -> Truy vấn: '{query}'")
        print(f"  -> Kết quả tìm kiếm: {retrieved_name}")
    except Exception as e:
        print(f"❌ Lỗi ở bước Retriever: {e}")
        return

    # 4. Test Generator
    print("\n[4/4] Test Generator (Local LLM Inference)...")
    try:
        generator = CodeGenerator()
        test_prefix = "def calculate_total(prices):\n    # Calculate sum using add_numbers\n"
        
        res = generator.generate_with_repocoder_loop(
            retriever=hybrid_retriever,
            prefix_code=test_prefix,
            max_iterations=1,
            top_k=1
        )
        
        print("\n" + "=" * 20 + " KẾT QUẢ ĐẦU RA " + "=" * 20)
        print(res.get("final_code", "Không có đầu ra"))
        print("=" * 56)
        print("\n✅ [SUCCESS] Pipeline chạy hoàn hảo trên Local Laptop!")

    except Exception as e:
        print(f"❌ Lỗi ở bước Generator: {e}")

# ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT BỊ THIẾU
if __name__ == "__main__":
    test_pipeline()