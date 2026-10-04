"""
Download BEIR SciFact (5,183 abstracts, 1,109 claims with relevance judgements).

    python scripts/download_scifact.py          # -> data/scifact/

Source: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
SciFact (Wadden et al., EMNLP 2020): claims CC BY 4.0, abstracts ODC-By 1.0 (from S2ORC).
BEIR packaging: Thakur et al., NeurIPS 2021 Datasets and Benchmarks.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


def main(dest: Path = Path("data")) -> None:
    with urllib.request.urlopen(URL, timeout=120) as resp:
        payload = resp.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: got {digest}, expected {SHA256}")
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(payload)).extractall(dest)
    print(f"SciFact extracted to {dest / 'scifact'} (sha256 verified)")


if __name__ == "__main__":
    main()
