from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="beomi/KoAlpaca-Polyglot-5.8B",
    local_dir="./models/koalpaca-5.8b",
    local_dir_use_symlinks=False
)