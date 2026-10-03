
import os
import requests
from tqdm import tqdm
import tarfile

# === Configuration ===
# Example: replace with the actual file URL
dataset_url = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
output_path = "phoenix2014t.tar.gz"
extract_dir = "./datasets_hdd"

# === Download function ===


def download_file(url, output_path, ):
    with requests.get(url, stream=True, ) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


# === Main execution ===
if __name__ == "__main__":
    print("Starting download...")
    download_file(dataset_url, output_path, )
    print("Download complete.")
