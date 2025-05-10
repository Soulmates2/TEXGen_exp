from huggingface_hub import hf_hub_download, snapshot_download
import shutil

snapshot_download(repo_id="Andyx/TEXGen", allow_patterns="texgen_v1.ckpt", local_dir="assets/checkpoints")
shutil.rmtree("assets/checkpoints/.cache")