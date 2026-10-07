from src.data_loader import RepoDataLoader

# Đường dẫn tới file CCEval
path = r"D:\DSR301m\cceval-main\cceval-main\data\crosscodeeval_data\python\line_completion.jsonl"

# Load dataset
data = RepoDataLoader.load_benchmark_dataset(path)

print("Number of samples:", len(data))

# Lấy sample đầu tiên
sample = data[0]

print("\n--- SAMPLE ---")
print("sample_id:", sample.get("sample_id"))
print("repo_name:", sample.get("repo_name"))
print("file_path:", sample.get("file_path"))

print("\n--- PROMPT ---")
print(sample.get("prompt", "")[:500])

print("\n--- GROUND TRUTH ---")
print(repr(sample.get("ground_truth")))

print("\n--- ORIGINAL GROUNDTRUTH ---")
print(repr(sample.get("groundtruth")))

print("\n--- RIGHT CONTEXT ---")
print(repr(sample.get("right_context", ""))[:500])

print("\n--- METADATA ---")
print(sample.get("metadata"))