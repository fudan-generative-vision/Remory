"""Download client-side tokenizer metadata without actor or memory weights."""
from pathlib import Path

def resolve_tokenizer(config, actor_path=None):
    """Client metadata only: connecting to an existing server never downloads weights."""
    if actor_path is not None:
        path = Path(actor_path).expanduser().resolve()
        if (path / "config.json").is_file():
            return path
        if str(actor_path) != config["target_model"]:
            raise ValueError("tokenizer path must match the server's actor")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(config["target_model"], revision=config["target_revision"],
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"]))
