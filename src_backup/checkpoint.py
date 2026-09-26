import os
import json
import shutil
from pathlib import Path
from kaggle_secrets import UserSecretsClient
from kaggle.api.kaggle_api_extended import KaggleApi

def backup_to_kaggle_dataset(
    target_dir: str, 
    dataset_id: str, 
    zip_name: str = "chroma_db_backup"
):
    """
    Nén thư mục target_dir (Vector DB hoặc Outputs) và push trực tiếp lên Kaggle Dataset.
    - dataset_id ví dụ: 'your_username/code-rag-checkpoint'
    """
    try:
        # Load API Credentials từ Kaggle Secrets
        user_secrets = UserSecretsClient()
        os.environ['KAGGLE_USERNAME'] = user_secrets.get_secret("KAGGLE_USERNAME")
        os.environ['KAGGLE_KEY'] = user_secrets.get_secret("KAGGLE_KEY")
        
        api = KaggleApi()
        api.authenticate()

        # Thư mục tạm dùng để upload
        upload_dir = Path("./kaggle_upload_temp")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Nén thư mục dữ liệu
        archive_path = upload_dir / zip_name
        shutil.make_archive(str(archive_path), 'zip', target_dir)
        print(f"[*] Compressed {target_dir} to {archive_path}.zip")

        # Tạo file metadata cho Dataset nếu chưa có
        meta_path = upload_dir / "dataset-metadata.json"
        if not meta_path.exists():
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
                version_notes="Auto checkpoint sync", 
                dir_mode="zip"
            )
            print(" [CHECKPOINT] Successfully updated Kaggle Dataset!")
        except Exception:
            api.dataset_create_new(str(upload_dir), dir_mode="zip")
            print(" [CHECKPOINT] Created new Kaggle Dataset & uploaded successfully!")

    except Exception as e:
        print(f"⚠️ [CHECKPOINT WARNING] Backup skipped or failed: {e}")