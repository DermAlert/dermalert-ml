import time
from pathlib import Path
import requests

from ..config import Config

API_URL = "https://api.isic-archive.com/api/v2/images/{id}/"
cfg = Config()                       # default config
cfg.cache_dir.mkdir(exist_ok=True)

def fetch(isic_id: str, cache_dir: Path | None = None) -> Path:
    """Baixa imagem do ISIC (3 tentativas) e devolve Path local."""
    cache_dir = cache_dir or cfg.cache_dir
    dest = cache_dir / f"{isic_id}.jpg"
    if dest.exists():
        return dest

    for _ in range(3):
        r = requests.get(API_URL.format(id=isic_id), timeout=30)
        if r.ok:
            path_download = r.json()['files']['full']['url']
            r = requests.get(path_download, timeout=30)
            if r.ok:
                dest.write_bytes(r.content)
                return dest
        time.sleep(2)
    raise RuntimeError(f"Falha ao baixar {isic_id}")
