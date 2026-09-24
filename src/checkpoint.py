import os
import json
import shutil
from pathlib import Path

def backup_to_kaggle_dataset(
    target_dir: str, 
    dataset_id: str, 
    zip_name: str = "checkpoint_backup"
):
    """
    Nén thư mục target_dir (Vector DB hoặc Outputs) và push trực tiếp lên Kaggle Dataset.
    - dataset_id ví dụ: 'your_username/code-rag-checkpoint'
    """
    if not dataset_id or "/" not in dataset_id:
        print("[CHECKPOINT] Skip backup: Invalid or missing dataset_id.")
        return

    try:
        from kaggle_secrets import UserSecretsClient
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Load API Credentials từ Kaggle Secrets
        user_secrets = UserSecretsClient()
        try:
            os.environ['KAGGLE_USERNAME'] = user_secrets.get_secret("KAGGLE_USERNAME")
            os.environ['KAGGLE_KEY'] = user_secrets.get_secret("KAGGLE_KEY")
        except Exception as sec_err:
            print(f"⚠️ [CHECKPOINT WARNING] Missing Kaggle Secrets (KAGGLE_USERNAME / KAGGLE_KEY): {sec_err}")
            return

        api = KaggleApi()
        api.authenticate()

        # DỌN DẸP THƯ MỤC TẠM TRƯỚC KHI TẠO ZIP MỚI (Tránh tích tụ file zip cũ)
        upload_dir = Path("./kaggle_upload_temp")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        target_path = Path(target_dir)
        if not target_path.exists():
            print(f"⚠️ [CHECKPOINT WARNING] Target directory '{target_dir}' does not exist.")
            return

        # Nén thư mục dữ liệu
        archive_base = upload_dir / zip_name
        shutil.make_archive(str(archive_base), 'zip', target_dir)
        print(f"[*] Compressed '{target_dir}' to '{archive_base}.zip'")

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
            print("🟢 [CHECKPOINT] Successfully updated Kaggle Dataset version!")
        except Exception:
            # Nếu Dataset chưa tồn tại trên tài khoản Kaggle, tạo mới
            api.dataset_create_new(str(upload_dir), dir_mode="zip", quiet=True)
            print("🟢 [CHECKPOINT] Created new Kaggle Dataset & uploaded successfully!")

    except Exception as e:
        print(f"⚠️ [CHECKPOINT WARNING] Backup skipped or failed: {e}")