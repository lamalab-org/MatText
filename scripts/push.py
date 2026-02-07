#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_file, list_repo_files, delete_file, CommitOperationAdd, CommitOperationDelete

def main():
    parser = argparse.ArgumentParser(description="Push a JSON file to a Hugging Face dataset repo as JSONL.")
    parser.add_argument("--json-path", required=True, help="Path to the input JSON file.")
    parser.add_argument("--token", required=True, help="Hugging Face access token.")
    parser.add_argument("--repo", required=True, help="Target dataset repo in the form 'namespace/repo_name' (e.g., 'lila-ai/mattext-results').")
    parser.add_argument("--path-in-repo", default="data/inference_results.jsonl", help="Path within the dataset repo for the JSONL (default: data/inference_results.jsonl).")
    parser.add_argument("--commit-message", default="Add/update inference results JSONL", help="Commit message.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    parser.add_argument("--branch", default=None, help="Target branch (e.g., 'main' or 'dev'). Default: repo default branch.")
    parser.add_argument("--append", action="store_true", help="Append as a new line rather than overwrite.")
    args = parser.parse_args()

    # Auth
    os.environ["HF_TOKEN"] = args.token
    #HfFolder.save_token(args.token)
    api = HfApi()

    # Ensure dataset repo exists
    try:
        create_repo(repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True, token=args.token)
    except Exception as e:
        # If it already exists or creation failed due to exist_ok, continue
        pass

    # Load and validate JSON
    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON path not found: {json_path}")

    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    # Optional light validation based on your screenshot structure
    required_top_level = ["task_name", "checkpoint", "num_folds", "folds_results", "summary"]
    missing = [k for k in required_top_level if k not in data]
    if missing:
        print(f"Warning: JSON missing expected keys: {missing}")

    # Prepare content as JSONL
    record_line = json.dumps(data, separators=(",", ":")) + "\n"

    # If appending, we need to fetch existing file content if it exists
    operations = []
    target_path = args.path_in_repo

    existing_files = set(api.list_repo_files(repo_id=args.repo, repo_type="dataset", revision=args.branch or "main" if args.branch else None))
    file_exists = target_path in existing_files

    if args.append and file_exists:
        # Download existing file
        local_tmp = Path(".hf_tmp_existing.jsonl")
        try:
            hf_hub_download(
                repo_id=args.repo,
                repo_type="dataset",
                filename=target_path,
                local_dir=".",
                local_dir_use_symlinks=False,
                revision=args.branch if args.branch else None,
                token=args.token
            )
            existing_content = Path(target_path).read_text(encoding="utf-8")
            merged = existing_content + record_line
            Path(local_tmp).write_text(merged, encoding="utf-8")
            upload_path = str(local_tmp)
        except Exception:
            # If download fails, fall back to creating the file fresh
            upload_path = None
        finally:
            # Clean the downloaded file copy if present
            try:
                Path(target_path).unlink(missing_ok=True)
            except Exception:
                pass

        if upload_path is None:
            # Could not fetch existing content; just create new with current line
            local_new = Path(".hf_tmp_new.jsonl")
            local_new.write_text(record_line, encoding="utf-8")
            upload_path = str(local_new)

        operations.append(
            CommitOperationAdd(path_in_repo=target_path, path_or_fileobj=upload_path)
        )
    else:
        # Overwrite or create new
        local_new = Path(".hf_tmp_new.jsonl")
        local_new.write_text(record_line, encoding="utf-8")
        operations.append(
            CommitOperationAdd(path_in_repo=target_path, path_or_fileobj=str(local_new))
        )

    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        operations=operations,
        commit_message=args.commit_message,
        revision=args.branch if args.branch else None,
        parent_commit=None,
        token=args.token,
    )
    # Cleanup temp files
    for p in [".hf_tmp_existing.jsonl", ".hf_tmp_new.jsonl"]:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()