"""Resumable HF Hub API upload for the extracted WorldFloods Zarr store."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import httpx
from huggingface_hub import CommitOperationAdd, HfApi


DEFAULT_ROOT = Path("/shared/home/rdelprete/.cache/hydranet/worldfloods")
DEFAULT_STATE = Path("/shared/home/rdelprete/.cache/hydranet/phisatnet-worldfloods-upload-state.json")
DEFAULT_REPO_ID = "sirbastiano94/phisatnet-worldfloods"
REPO_TYPE = "dataset"
EXPECTED_TRAINVAL_FILES_PER_SAMPLE = 24
VERIFY_CHUNK = 500


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def load_state(path: Path) -> dict:
    if path.exists():
        state = json.loads(path.read_text())
    else:
        state = {}
    state.setdefault("done", [])
    state.setdefault("attempting", None)
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def metadata_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for rel in [
        ".zgroup",
        ".zattrs",
        "test/.zgroup",
        "test/.zattrs",
        "trainval/.zgroup",
        "trainval/.zattrs",
        "README.md",
    ]:
        path = root / rel
        if path.is_file():
            paths.append(path)
    return paths


def sample_paths(root: Path, split: str, sample_id: str, expected_file_count: int | None = None) -> list[Path]:
    base = root / split / sample_id
    if not base.is_dir():
        raise FileNotFoundError(base)

    paths = sorted(path for path in base.rglob("*") if path.is_file())
    if expected_file_count is not None and len(paths) != expected_file_count:
        raise RuntimeError(f"{split}/{sample_id} has {len(paths)} files, expected {expected_file_count}")
    return paths


def trainval_batch_paths(root: Path, start: int, end: int) -> list[Path]:
    paths: list[Path] = []
    for idx in range(start, end + 1):
        paths.extend(
            sample_paths(
                root,
                "trainval",
                f"{idx:07d}",
                expected_file_count=EXPECTED_TRAINVAL_FILES_PER_SAMPLE,
            )
        )
    return paths


def rel_paths(root: Path, paths: Iterable[Path]) -> list[str]:
    return [path.relative_to(root).as_posix() for path in paths]


def operations(root: Path, paths: list[Path]) -> list[CommitOperationAdd]:
    return [
        CommitOperationAdd(path_in_repo=path.relative_to(root).as_posix(), path_or_fileobj=str(path))
        for path in paths
    ]


def size_gib(paths: list[Path]) -> float:
    return sum(path.stat().st_size for path in paths) / 1024**3


def remote_has_all(api: HfApi, repo_id: str, root: Path, paths: list[Path]) -> bool:
    expected = rel_paths(root, paths)
    seen: set[str] = set()

    for idx in range(0, len(expected), VERIFY_CHUNK):
        infos = api.get_paths_info(repo_id=repo_id, repo_type=REPO_TYPE, paths=expected[idx : idx + VERIFY_CHUNK])
        seen.update(getattr(info, "path", "") for info in infos)

    missing = [path for path in expected if path not in seen]
    if missing:
        log(f"remote verification missing {len(missing)} files, first={missing[0]}")
        return False
    return True


def upload_key(
    *,
    api: HfApi,
    root: Path,
    repo_id: str,
    state_path: Path,
    state: dict,
    done: set[str],
    key: str,
    paths: list[Path],
    commit_message: str,
    num_threads: int,
) -> None:
    if key in done:
        log(f"skip {key}")
        return
    if not paths:
        raise RuntimeError(f"{key} resolved to no files")

    if state.get("attempting") == key or state.get("verify_before_upload") == key:
        log(f"check remote {key}")
        if remote_has_all(api, repo_id, root, paths):
            log(f"remote already has {key}")
            done.add(key)
            state["done"] = sorted(done)
            state["attempting"] = None
            state.pop("verify_before_upload", None)
            save_state(state_path, state)
            return

    state["attempting"] = key
    save_state(state_path, state)

    attempt = 0
    while True:
        attempt += 1
        log(f"upload {key}: {len(paths)} files, {size_gib(paths):.3f} GiB, attempt {attempt}")
        started = time.time()
        try:
            info = api.create_commit(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                operations=operations(root, paths),
                commit_message=commit_message,
                num_threads=num_threads,
            )
        except Exception as exc:
            log(f"exception during {key}: {type(exc).__name__}: {exc}")
            wait = 45 if isinstance(exc, httpx.TimeoutException) else min(180, 30 * attempt)
            log(f"waiting {wait}s before remote verification for {key}")
            time.sleep(wait)
            try:
                if remote_has_all(api, repo_id, root, paths):
                    log(f"done {key}: verified on remote after exception")
                    done.add(key)
                    state["done"] = sorted(done)
                    state["attempting"] = None
                    state.pop("verify_before_upload", None)
                    save_state(state_path, state)
                    return
            except Exception as verify_exc:
                log(f"verification failed for {key}: {type(verify_exc).__name__}: {verify_exc}")

            if attempt >= 4:
                state["verify_before_upload"] = key
                save_state(state_path, state)
                raise

            sleep_s = min(300, 45 * attempt)
            log(f"retrying {key} after {sleep_s}s")
            time.sleep(sleep_s)
            continue

        log(f"done {key}: {getattr(info, 'commit_url', '') or getattr(info, 'oid', '')} in {time.time() - started:.1f}s")
        done.add(key)
        state["done"] = sorted(done)
        state["attempting"] = None
        state.pop("verify_before_upload", None)
        save_state(state_path, state)
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--trainval-first", type=int, default=0)
    parser.add_argument("--trainval-last", type=int, default=61622)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=REPO_TYPE, exist_ok=True)

    state = load_state(args.state)
    done = set(state["done"])

    upload_key(
        api=api,
        root=args.root,
        repo_id=args.repo_id,
        state_path=args.state,
        state=state,
        done=done,
        key="metadata",
        paths=metadata_paths(args.root),
        commit_message="Upload metadata",
        num_threads=args.num_threads,
    )

    if not args.skip_test:
        for idx in range(18):
            sample_id = f"{idx:07d}"
            key = f"test/{sample_id}"
            paths = [] if key in done else sample_paths(args.root, "test", sample_id)
            upload_key(
                api=api,
                root=args.root,
                repo_id=args.repo_id,
                state_path=args.state,
                state=state,
                done=done,
                key=key,
                paths=paths,
                commit_message=f"Upload test/{sample_id}",
                num_threads=args.num_threads,
            )

    log(
        f"trainval samples: {args.trainval_last - args.trainval_first + 1}, "
        f"min={args.trainval_first:07d}, max={args.trainval_last:07d}, batch_size={args.batch_size}"
    )
    start = args.trainval_first
    while start <= args.trainval_last:
        end = min(start + args.batch_size - 1, args.trainval_last)
        key = f"trainval/{start:07d}-{end:07d}"
        paths = [] if key in done else trainval_batch_paths(args.root, start, end)
        upload_key(
            api=api,
            root=args.root,
            repo_id=args.repo_id,
            state_path=args.state,
            state=state,
            done=done,
            key=key,
            paths=paths,
            commit_message=f"Upload trainval {start:07d}-{end:07d}",
            num_threads=args.num_threads,
        )
        start = end + 1

    log(f"uploaded all: {len(done)} completed keys recorded")


if __name__ == "__main__":
    main()
