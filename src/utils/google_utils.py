import gdown
import tempfile
import re

def convert_to_direct_link(original_url):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)/', original_url)
    if not match:
        raise ValueError("Invalid Google Drive URL format. Unable to extract file ID.")
    file_id = match.group(1)
    return f"https://drive.google.com/uc?id={file_id}"

def download_weights(original_url):
    direct_link = convert_to_direct_link(original_url)
    temp_dir = tempfile.mkdtemp()
    local_path = f"{temp_dir}/model_weights.pt"
    try:
        gdown.download(direct_link, local_path, quiet=False)
        print(f"Model weights downloaded to: {local_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download weights: {e}")
    return local_path
