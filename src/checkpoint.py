import os
import json
import shutil
from pathlib import Path


def is_kaggle_environment() -> bool:
    """Kiểm tra xem code có đang chạy trong môi trường Kaggle Notebooks hay không."""
    return os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def backup_to_kaggle_dataset(
    target_dir: str, 
    dataset_id: str, 
    zip_name: str = "checkpoint_backup"
):
    """
    Nén thư mục target_dir (Vector DB hoặc Outputs) và push trực tiếp lên Kaggle Dataset.
    - An toàn khi chạy trên Google Colab / Local: Nếu không ở Kaggle hoặc không có credentials,
      sẽ tự động bỏ qua (skip) nhẹ nhàng mà không gây lỗi/crash luồng đánh giá.
    - dataset_id ví dụ: 'your_username/code-rag-checkpoint'
    """
    if not dataset_id or "/" not in dataset_id:
        return

    # Nếu không phải Kaggle và không có API key Kaggle thiết lập sẵn, bỏ qua an toàn
    has_kaggle_env_key = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    
    if not is_kaggle_environment() and not has_kaggle_env_key and not kaggle_json_path.exists():
        # Đang chạy trên Colab/Local không cấu hình Kaggle API -> Bỏ qua không in cảnh báo spam
        return

    try:
        # 1. Thử lấy credential từ Kaggle Secrets nếu đang trên Kaggle
        if is_kaggle_environment():
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                os.environ['KAGGLE_USERNAME'] = user_secrets.get_secret("KAGGLE_USERNAME")
                os.environ['KAGGLE_KEY'] = user_secrets.get_secret("KAGGLE_KEY")
            except Exception:
                # Nếu không set Secrets trên Kaggle, thử dùng biến môi trường hoặc file kaggle.json có sẵn
                pass

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        # DỌN DẸP THƯ MỤC TẠM TRƯỚC KHI TẠO ZIP MỚI (Tránh tích tụ file zip cũ)
        upload_dir = Path("./kaggle_upload_temp")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        target_path = Path(target_dir)
        if not target_path.exists():
            return

        # Nén thư mục dữ liệu
        archive_base = upload_dir / zip_name
        shutil.make_archive(str(archive_base), 'zip', target_dir)
        print(f"[*] [Kaggle Backup] Compressed '{target_dir}' to '{archive_base}.zip'")

        # Tạo file metadata cho Dataset
        meta_path = upload_dir / "dataset-metadata.json"
        meta_data = {
            "title": dataset_id.split("/")[-1],
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        # Upload lên Kaggle Dataset
        try:
            api.dataset_create_version(
                str(upload_dir), 
                version_notes=f"Auto checkpoint sync: {zip_name}", 
                dir_mode="zip"
            )
            print(f"🟢 [CHECKPOINT] Successfully updated Kaggle Dataset: {dataset_id}")
        except Exception:
            # Nếu Dataset chưa tồn tại trên tài khoản Kaggle, tạo mới
            api.dataset_create_new(str(upload_dir), dir_mode="zip", quiet=True)
            print(f"🟢 [CHECKPOINT] Created new Kaggle Dataset & uploaded: {dataset_id}")

    except Exception as e:
        # Log nhẹ nhàng để không làm gián đoạn vòng lặp benchmark
        print(f"[Checkpoint Note] Kaggle sync skipped ({e})")