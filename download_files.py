import os
import subprocess
import sys

embedder_name = sys.argv[2]

hugg_link = "https://huggingface.co/Politrees/RVC_resources/resolve/main"
file_links = {
    "rmvpe.pt": f"{hugg_link}/predictors/rmvpe.pt",
    "hubert_base.pt": f"{hugg_link}/embedders/{embedder_name}.pt",
}

for file, link in file_links.items():
    file_path = os.path.join(file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(["wget", "-O", file_path, link], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Ошибка установки {file}: {e}")
