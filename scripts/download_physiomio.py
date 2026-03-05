from huggingface_hub import snapshot_download
import shutil
import os

if __name__ == "__main__":
    target_dir = "datasets/raw/physiomio"

    print("Downloading PhysioMio dataset...")
    
    snapshot_download(
        repo_id="formove-ai/physiomio",
        repo_type="dataset",
        local_dir=target_dir,
    )

    # Move only data contents up one level
    data_path = os.path.join(target_dir, "data")

    if os.path.exists(data_path):
        for item in os.listdir(data_path):
            shutil.move(
                os.path.join(data_path, item),
                os.path.join(target_dir, item)
            )
        shutil.rmtree(data_path)

    print("Raw PhysioMio has been downloaded and is in the 'datasets/raw/physiomio' directory.")