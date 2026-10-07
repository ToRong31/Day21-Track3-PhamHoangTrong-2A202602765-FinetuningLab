"""Audit frozen B2 artifacts without changing training data or measured scores."""
import collections
import hashlib
import json
import pathlib
import re
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "bonus/B2_UIT/data"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main():
    from prepare_uit_b2 import SCENARIOS, SERVICES, URGENCY, SENTIMENT

    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    checksums = json.loads((DATA / "checksums.json").read_text(encoding="utf-8"))
    findings = []
    reviewed = []
    split_records = {}
    hashes = {}
    for split, expected in [("train_seed", 240), ("eval_target", 48)]:
        path = DATA / (split + ".jsonl")
        records = rows(path)
        split_records[split] = records
        assert len(records) == expected
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[path.name][:16] == checksums[path.name]
        assert records == rows(ROOT / "data/uit_b2" / path.name), "Local source differs from measured corpus"
        entries = [entry for entry in manifest if entry["split"] == split]
        assert len(entries) == len(records)
        for record, entry in zip(records, entries):
            intent, number, variant = entry["id"].rsplit("-", 2)
            assert entry["scenario_id"] == intent + "-" + number
            assert (int(number) < 10) == (split == "train_seed")
            label = record["label"]
            assert set(label) == {"intent", "urgency", "product", "sentiment"}
            assert json.loads(record["output"]) == label == entry["label"]
            assert label["intent"] == intent
            service = re.search(r"Dịch vụ: ([^.]+)\.", record["input"])
            assert service and service.group(1) == label["product"] == SERVICES[intent]
            scenario = SCENARIOS[intent][int(number)]
            assert scenario in record["input"]
            urgency = [key for key, cues in URGENCY.items() if any(cue in record["input"] for cue in cues)]
            sentiment = [key for key, cues in SENTIMENT.items() if any(cue in record["input"] for cue in cues)]
            assert urgency == [label["urgency"]]
            assert sentiment == [label["sentiment"]]
            reviewed.append({"id": entry["id"], "scenario_id": entry["scenario_id"],
                             "split": split, "label_checks": "pass"})
    train = split_records["train_seed"]
    target = split_records["eval_target"]
    assert len({normalized(r["input"]) for r in train + target}) == 288
    train_ids = {r["scenario_id"] for r in reviewed if r["split"] == "train_seed"}
    target_ids = {r["scenario_id"] for r in reviewed if r["split"] == "eval_target"}
    assert not train_ids & target_ids
    for name in ["eval_regression.jsonl", "holdout_secret.jsonl"]:
        path = DATA / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[name][:16] == checksums[name]
        assert not {normalized(r["input"]) for r in train} & {
            normalized(r.get("input", r.get("instruction", ""))) for r in rows(path)}
    train_split, val_split = rows(DATA / "split/train.jsonl"), rows(DATA / "split/val.jsonl")
    assert len(train_split) == 216 and len(val_split) == 24
    identity = lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True)
    assert collections.Counter(map(identity, train_split + val_split)) == collections.Counter(map(identity, train))
    scenario_by_input = {r["input"]: entry["scenario_id"] for r, entry in
                         zip(train, [m for m in manifest if m["split"] == "train_seed"])}
    shared_validation = {scenario_by_input[r["input"]] for r in train_split} & {
        scenario_by_input[r["input"]] for r in val_split}
    findings.extend([
        {"severity": "limitation", "ids": ["hoc_vu-00", "hoc_vu-10"],
         "detail": "Training asks about preserving study results; evaluation asks about temporary leave. Semantically related despite distinct scenario IDs."},
        {"severity": "limitation", "ids": ["hoc_phi-02", "hoc_phi-08", "giay_to-03", "giay_to-11"],
         "detail": "Document corrections route by service context: fee receipts to hoc_phi; student certificates to giay_to. Context must be retained."},
        {"severity": "limitation", "ids": ["dang_ky_mon-08", "lich_hoc-08"],
         "detail": "Conflicting class registration routes to dang_ky_mon; conflicting exam schedule routes to lich_hoc."},
        {"severity": "limitation", "detail": "Service names reveal intent; urgency and sentiment use explicit repeated cues. Target accuracy measures this synthetic task, not real support quality."},
        {"severity": "limitation", "detail": "Positive sentiment describes prior support even when a current issue exists; label follows explicit expressed sentiment, not inferred frustration."},
    ])
    result = {"review_date": "2026-10-07", "reviewer": "Codex AI assistant",
              "method": "AI semantic review of all 72 scenario texts and label policy; exhaustive deterministic checks of all 288 instantiated rows. Not independent human annotation.",
              "reviewed_rows": len(reviewed), "reviewed_scenarios": len(train_ids | target_ids),
              "label_errors_found": 0, "data_rows_changed": 0, "dataset_sha256": hashes,
              "train_validation_shared_scenarios": len(shared_validation),
              "findings": findings, "row_checks": reviewed,
              "conclusion": "Labels are consistent with the declared synthetic taxonomy. B2 evidence prepared for grading with stated limitations; no human review or awarded points claimed."}
    out = DATA / "SEMANTIC_REVIEW.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for directory in [DATA, ROOT / "data/uit_b2"]:
        quality_path = directory / "quality_checks.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality.update({"review_status": "AI semantic review complete; independent human review not recorded.",
                        "reviewer": "Codex AI assistant", "review_date": result["review_date"],
                        "reviewed_rows": 288, "label_errors_found": 0, "data_rows_changed": 0,
                        "train_validation_shared_scenarios": len(shared_validation),
                        "semantic_review_file": "SEMANTIC_REVIEW.json"})
        quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (directory / "SEMANTIC_REVIEW.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"row_checks", "dataset_sha256", "findings"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
