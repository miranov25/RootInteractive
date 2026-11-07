"""Local, pinned copies of the external scripts every RootInteractive page requests (proposal v0.2 §2, §5.1).

A RootInteractive page loads pako, vis and onnxruntime-web from the internet; onnxruntime-web without a
version. The engine serves pinned copies instead, so a test never depends on the network or on an
outside release. The files are fetched once from the npm registry, verified against the registry's
integrity hash (sha512 of the package) and against the sha256 of every extracted file, and stored in
headless/vendor/ (git-ignored; only VENDOR.md is committed).

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.engine.vendor fetch
    python -m RootInteractive.InteractiveDrawing.bokeh.headless.engine.vendor check
"""
import base64
import hashlib
import io
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"


@dataclass(frozen=True)
class Pin:
    package: str
    version: str
    tarball: str
    integrity: str                             # npm dist.integrity of the tarball
    url_glob: str                              # what the page requests; routed to the first file
    files: Tuple[Tuple[str, str, str], ...]    # (local name, path inside the tarball, sha256)

    @property
    def main(self) -> Path:
        return VENDOR_DIR / self.files[0][0]


PINS: Tuple[Pin, ...] = (
    Pin("pako", "2.0.2", "https://registry.npmjs.org/pako/-/pako-2.0.2.tgz",
        "sha512-9e8DRI3+dRLomCmMBAH30B2ejh+blwXr7VmMEx/pVFZlSDA7oyI8uKMhKXr8IrZpoxBF2YlxUvhqRXzTT1i0NA==",
        "**/pako.min.js",
        (("pako-2.0.2.min.js", "package/dist/pako.min.js", "0af0740eec63783df99a0dca4baed7b6377cd009ad049b644073bf3ed6c0832f"),
         ("pako-2.0.2.LICENSE", "package/LICENSE", "a04665b3b2de56c66730c1f720f528175739e4104f79073614aa611da1e85539"))),
    Pin("vis", "4.16.1", "https://registry.npmjs.org/vis/-/vis-4.16.1.tgz",
        "sha512-YcTQhKo8+kCqWaKoLkXyrGik8UC93TYrqNdwZm5bUN130ouunvjJyzvu43nXEynNKnh+d72OGFXtD86OukZqkA==",
        "**/vis.min.js",
        (("vis-4.16.1.min.js", "package/dist/vis.min.js", "7b461b95eaf9aedbb6e4765a2913a75ffe086d8c5b45d12c18a96fd43c55edc4"),
         ("vis-4.16.1.LICENSE-MIT", "package/LICENSE-MIT", "789b37de42c7d3c6a1ce3f0412e93d763148c9bf8b44b8936f447ea4c35dcb2b"),
         ("vis-4.16.1.LICENSE-APACHE-2.0", "package/LICENSE-APACHE-2.0",
          "1509904280a272780fd69c04b858f512cf216f8e3464a0dd98d8f34416feced9"))),
    # the newest release older than two weeks at stage 4.2a (2026-11-07); MIT, notice in the file header
    Pin("onnxruntime-web", "1.30.0", "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.30.0.tgz",
        "sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ==",
        "**/ort.min.js",
        (("onnxruntime-web-1.30.0.ort.min.js", "package/dist/ort.min.js",
          "7ae472a8ec976bcb69ad46b89b0a6ec8f3c5dbea4477674a26958672d1f3322d"),)),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> List[str]:
    """Problems with the local copies; an empty list means every file is present with its pinned hash."""
    problems = []
    for pin in PINS:
        for name, _, sha in pin.files:
            p = VENDOR_DIR / name
            if not p.exists():
                problems.append(f"missing {p.name} ({pin.package} {pin.version})")
            elif _sha256(p) != sha:
                problems.append(f"hash mismatch {p.name}: expected sha256 {sha}")
    return problems


def fetch(verbose: bool = True) -> None:
    """Download every pinned package, verify the tarball and each extracted file, write them to vendor/."""
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    for pin in PINS:
        data = urllib.request.urlopen(pin.tarball, timeout=120).read()
        algo, b64 = pin.integrity.split("-", 1)
        if base64.b64encode(hashlib.new(algo, data).digest()).decode() != b64:
            raise RuntimeError(f"{pin.package} {pin.version}: tarball integrity mismatch")
        tf = tarfile.open(fileobj=io.BytesIO(data))
        for name, inner, sha in pin.files:
            blob = tf.extractfile(inner).read()
            if hashlib.sha256(blob).hexdigest() != sha:
                raise RuntimeError(f"{pin.package} {pin.version}: {inner} sha256 mismatch")
            (VENDOR_DIR / name).write_bytes(blob)
            if verbose:
                print(f"vendor: {name} ({len(blob)} bytes) verified")


def routes() -> Dict[str, Path]:
    """URL glob -> local file, used by the browser runner."""
    return {pin.url_glob: pin.main for pin in PINS}


def versions() -> Dict[str, str]:
    return {pin.package: pin.version for pin in PINS}


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "check") == "fetch":
        fetch()
    issues = check()
    print("vendor: OK, every pinned file present and verified" if not issues else "vendor: " + "; ".join(issues))
    sys.exit(1 if issues else 0)
