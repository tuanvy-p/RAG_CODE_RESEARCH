import torch
from sentence_transformers import SentenceTransformer

# 1. Kiểm tra GPU CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy trên thiết bị: {device}")

# 2. Tải mô hình BAAI/bge-m3 lên GPU RTX 5060
model = SentenceTransformer("BAAI/bge-m3", device=device)

# 3. Chạy thử tạo embedding cho mã nguồn
sample_code = ["def add(a, b): return a + b", "class CodeEvaluator: pass"]
embeddings = model.encode(sample_code, batch_size=32, show_progress_bar=True)

print(f"Số lượng vector: {len(embeddings)}")
print(f"Kích thước mỗi vector (Dimension): {len(embeddings[0])}") # Mặc định 1024