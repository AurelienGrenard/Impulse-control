"""Create a materialized ZIP snapshot for the manuscript supplementary materials."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "Impulse-control-reproducibility"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def sha256(path: Path) -> str:
    """Compute the SHA-256 digest of one artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files() -> list[Path]:
    """List tracked and intentional untracked files, honoring `.gitignore`."""
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    paths = [ROOT / name for name in output.decode().split("\0") if name]
    return sorted(path for path in paths if path.is_file())


def main() -> None:
    """Write a ZIP containing real checkpoint bytes and an internal checksum manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "Impulse-control-reproducibility.zip",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    files = artifact_files()
    for path in files:
        if path.suffix == ".pt":
            with path.open("rb") as stream:
                if stream.read(len(LFS_HEADER)) == LFS_HEADER:
                    raise RuntimeError(
                        f"Git LFS pointer found instead of checkpoint data: {path}"
                    )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = [f"git_commit  {commit}"]
    manifest.extend(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{ARCHIVE_ROOT}/{relative}")
        archive.writestr(
            f"{ARCHIVE_ROOT}/ARCHIVE_MANIFEST.txt", "\n".join(manifest) + "\n"
        )
    print(f"archive: {output}")
    print(f"sha256: {sha256(output)}")


if __name__ == "__main__":
    main()
