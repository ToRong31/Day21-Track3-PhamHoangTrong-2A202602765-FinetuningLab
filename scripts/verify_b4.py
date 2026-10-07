"""Verify the imported B4 evidence against its original per-rank artifacts."""
import csv
import hashlib
import io
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "bonus/B4/B4_evidence.zip"
if not ARCHIVE.exists():
    ARCHIVE = ROOT / "submission/B4_evidence.zip"  # Older evidence bundles.
DEST = ROOT / "bonus/B4"


def main():
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        for name in names:
            path = pathlib.PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
            assert ":" not in name
        def read(name):
            return archive.read(name).decode("utf-8-sig")
        def document(name):
            return json.loads(read(name))
        def records(name):
            return [json.loads(line) for line in read(name).splitlines() if line.strip()]
        submitted = document("comparison.json")
        declared_hashes = document("data_hashes.json")
        expected_files = list(declared_hashes) + [
            "requirements.txt", "notebooks/03_train_correct.py", "notebooks/05_evaluate_and_verdict.py"]
        reference = None
        config_reference = None
        summary = []
        for rank in (8, 16, 64):
            prefix = f"r{rank}/"
            runs = list(csv.DictReader(io.StringIO(read(prefix + "results/runs.csv"))))
            assert len(runs) == 1
            run = runs[0]
            config = document(prefix + "adapters/correct/adapter_config.json")
            assert int(run["r"]) == config["r"] == rank
            assert int(run["lora_alpha"]) == config["lora_alpha"] == 2 * rank
            assert int(run["max_steps"]) == 30 and float(run["learning_rate"]) == 0.0001
            assert run["placement"] == "text-linear" and run["mask_mode"] == "assistant-only"
            assert run["model"] == config["base_model_name_or_path"] == "unsloth/Qwen3.5-4B"
            assert run["precision"] == "fp16" and run["load_in_4bit"] == "False"
            comparable = {k: v for k, v in config.items() if k not in {"r", "lora_alpha", "target_modules"}}
            comparable["target_modules"] = sorted(config["target_modules"])
            if config_reference is None:
                config_reference = comparable
            assert comparable == config_reference
            expected_files_rank = expected_files + [name[len(prefix):] for name in names
                if name.startswith(prefix + "src/labkit/")]
            hashes = {name: hashlib.sha256(archive.read(prefix + name)).hexdigest()
                      for name in expected_files_rank}
            if reference is None:
                reference = hashes
            assert hashes == reference, "Code, data or baseline differ between ranks"
            assert all(hashes[name] == digest for name, digest in declared_hashes.items())
            for name, count in [("data/split/train.jsonl", 225), ("data/split/val.jsonl", 25),
                                ("data/eval_target.jsonl", 50), ("data/eval_regression.jsonl", 15)]:
                imported = records(prefix + name)
                assert len(imported) == count
                local = [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]
                assert imported == local, "Imported data differs from core"
            baseline = document(prefix + "results/baselines_frozen.json")
            assert baseline == json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
            verdict = document(prefix + "results/verdict.json")
            score = verdict["comparison"][2]
            assert score["n"] == 50
            entry = next(item for item in submitted if item["rank"] == rank)
            assert entry["training"] == run and entry["scores"] == score
            summary.append({"rank": rank, "alpha": config["lora_alpha"], **score,
                            "passed": verdict["verdict"]["passed"], "training": run,
                            "source": "bonus/B4/" + prefix + "results/verdict.json"})
        for name in names:
            out = DEST.joinpath(*pathlib.PurePosixPath(name).parts)
            content = archive.read(name)
            if out.exists():
                assert out.read_bytes() == content, "Existing evidence differs; preserve it"
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(content)
    result = {"status": "verified_from_original_colab_artifacts", "rows": summary,
              "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
              "matching_code_data_baseline_sha256": reference,
              "target_effect_ranges": {"learning_rate": 0.97, "rank": 0.135, "placement": 0.0},
              "limitations": ["Environment snapshot was captured after all runs, not independently per run.",
                              "Model revision is null in adapter configs; immutable base commit not recorded.",
                              "Placement/LR comparisons reuse the earlier core session; no repeat/variance estimate.",
                              "This evidence bundle contains adapter configs, not sweep weights."]}
    (ROOT / "results/b4_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("B4 verified: 3 ranks; matching code/data/baseline; 225 train, 50 target, 15 regression.")
    for row in summary:
        print(f"r={row['rank']}: target={row['target']}, regression={row['regression']}, VRAM={row['training']['peak_vram_gb']} GB")


if __name__ == "__main__":
    main()
