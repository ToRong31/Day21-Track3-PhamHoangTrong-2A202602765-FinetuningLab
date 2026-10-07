"""Publish only the measured main adapter and verify anonymous Hub access."""
import argparse
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE = ROOT / "bonus/B5_HUB"
FILES = ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
         "tokenizer_config.json", "tokenizer.json"]


def main():
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import RepositoryNotFoundError

    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise SystemExit("HuggingFace login required; run hf auth login locally.")
    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = username + "/lab21-qwen35-triage-vi"
    print("Authenticated account:", username, flush=True)
    print("Destination:", repo_id, flush=True)
    config = json.loads((ROOT / "adapters/correct/adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == "unsloth/Qwen3.5-4B"
    assert config["r"] == 16 and config["lora_alpha"] == 32
    for name in FILES:
        shutil.copy2(ROOT / "adapters/correct" / name, STAGE / name)
    for name in ["verdict.json", "runs.csv", "merge_check.json", "baselines_frozen.json"]:
        dest = STAGE / "evaluation" / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "results" / name, dest)
    card = STAGE / "README.md"
    text = card.read_text(encoding="utf-8").replace("REPLACE_WITH_THIS_REPOSITORY_ID", repo_id)
    card.write_text(text, encoding="utf-8")
    hashes = {p.relative_to(STAGE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in STAGE.rglob("*") if p.is_file() and ".cache" not in p.parts}
    if not args.publish:
        print("Prepared", len(hashes), "files. Use --publish to upload.")
        return
    try:
        existing = api.model_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="model", private=False)
    else:
        if existing.private:
            raise SystemExit("Destination already exists privately; inspect before changing visibility.")
        remote_card = existing.card_data.to_dict() if existing.card_data else {}
        if remote_card.get("base_model") != config["base_model_name_or_path"]:
            raise SystemExit("Existing repository has a different base; refusing to replace it.")
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(STAGE),
                               allow_patterns=list(hashes),
                               commit_message="Publish Lab21 measured Vietnamese ticket-triage LoRA adapter")
    anonymous = HfApi(token=False).model_info(repo_id, revision=commit.oid, files_metadata=True)
    assert not anonymous.private
    remote = {entry.rfilename: entry for entry in anonymous.siblings}
    assert set(hashes).issubset(remote)
    weights = remote["adapter_model.safetensors"]
    assert weights.lfs and weights.lfs.sha256 == hashes["adapter_model.safetensors"]
    assert weights.size == (STAGE / "adapter_model.safetensors").stat().st_size
    receipt = {"repo_id": repo_id, "url": "https://huggingface.co/" + repo_id,
               "commit_sha": commit.oid, "private": False, "anonymous_access_verified": True,
               "adapter_weights_sha256_verified": True, "published_on": "2026-10-07",
               "source_adapter": "adapters/correct", "files_sha256": hashes}
    (ROOT / "results/hub_check.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
