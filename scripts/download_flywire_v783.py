"""Download official FlyWire v783 graph inputs and verify Zenodo checksums."""
import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://zenodo.org/records/10676866/files/"
FILES = {
    "proofread_root_ids_783.npy": ("e0e6c19732fd8c7a4e39a2d170105421", 1114168),
    "proofread_connections_783.feather": ("f48f972d262323a102aed49af1396b8a", 852022274),
}


def verify(path: Path, md5: str, size: int) -> bool:
    if path.stat().st_size != size:
        return False
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == md5


def download(destination: Path, filename: str, md5: str, size: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / filename
    partial = destination / (filename + ".part")
    if final.exists() and verify(final, md5, size):
        print(f"Already verified: {final}")
        return
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        BASE + filename + "?download=1",
        headers={"User-Agent": "Pretorius-Connectome-research/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if not verify(partial, md5, size):
            raise ValueError(f"Integrity verification failed: {filename}")
        partial.replace(final)
        print(f"Verified: {final}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/flywire_v783"))
    args = parser.parse_args()
    for filename, (md5, size) in FILES.items():
        download(args.destination, filename, md5, size)


if __name__ == "__main__":
    main()
