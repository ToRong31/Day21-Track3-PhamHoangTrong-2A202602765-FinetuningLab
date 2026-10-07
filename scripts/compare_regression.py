"""Supplementary per-question regression comparison; preserves frozen metrics."""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path.cwd()
assert (ROOT / "results/baselines_frozen.json").exists(), "Run from the lab repo."
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate
from labkit.config import get_tier
from peft import PeftModel

frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
tier = get_tier("T4")
assert tier.model_id == frozen["model"], "Base model mismatch."
data_path = ROOT / "data/eval_regression.jsonl"
checksums = json.loads((ROOT / "data/checksums.json").read_text(encoding="utf-8"))
expected = checksums.get("eval_regression.jsonl")
assert expected and hashlib.sha256(data_path.read_bytes()).hexdigest()[:16] == expected, "Eval changed."
samples = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(samples) == frozen["n_regression"], "Eval slice mismatch."
prompts = [row["instruction"] for row in samples]

# NB2 and NB5 use no system prompt and max_new_tokens=96 for regression.
model, tok = generate.load_base(tier)
model.eval()
base_preds, _ = generate.generate_batch(model, tok, prompts, system=None,
                                       max_new_tokens=96, label="base/regression supplementary")
del model
generate.free_memory()
model, tok = generate.load_base(tier)
model = PeftModel.from_pretrained(model, str(ROOT / "adapters/correct"))
model.eval()
ft_preds, _ = generate.generate_batch(model, tok, prompts, system=None,
                                     max_new_tokens=96, label="correct/regression supplementary")
del model
generate.free_memory()

rows = []
for i, (sample, base, ft) in enumerate(zip(samples, base_preds, ft_preds)):
    sb = ev.keyword_recall(base, sample["keywords"])
    sf = ev.keyword_recall(ft, sample["keywords"])
    rows.append({"id": i, "instruction": sample["instruction"],
                 "keywords": sample["keywords"], "baseline_pred": base,
                 "ft_pred": ft, "baseline_score": sb, "ft_score": sf,
                 "delta": sf - sb,
                 "comparison": "THUA" if sf < sb else "THANG" if sf > sb else "HOA"})
losses = sorted([r for r in rows if r["comparison"] == "THUA"], key=lambda r: r["delta"])
selected = losses[:5]
if len(selected) < 5:
    selected_ids = {r["id"] for r in selected}
    selected += [r for r in rows if r["id"] not in selected_ids][:5-len(selected)]
payload = {"note": "Supplementary regression measurement; frozen verdict unchanged.",
           "scorer": "keyword_recall", "system_prompt": None, "max_new_tokens": 96,
           "baseline_score": sum(r["baseline_score"] for r in rows)/len(rows),
           "ft_score": sum(r["ft_score"] for r in rows)/len(rows),
           "n_losses": len(losses), "rows": rows, "selected_examples": selected}
out = ROOT / "results/regression_comparison.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = ["# Ví dụ regression — phép đo bổ sung", "",
         "Đây là câu hỏi phổ thông, không phải ticket target. Điểm là keyword recall; "
         "cần đọc câu trả lời để phân biệt lỗi nội dung với khác cách diễn đạt.", "",
         f"Base: {payload['baseline_score']:.4f}; fine-tune: {payload['ft_score']:.4f}; "
         f"số ca thua: {len(losses)}.", ""]
for r in selected:
    lines += [f"## Câu {r['id']} — {r['comparison']}", "", r["instruction"], "",
              "Từ khóa chấm: " + ", ".join(r["keywords"]), "",
              f"Baseline ({r['baseline_score']:.4f}):", "", "```text", r["baseline_pred"], "```", "",
              f"Fine-tune ({r['ft_score']:.4f}):", "", "```text", r["ft_pred"], "```", ""]
md = ROOT / "results/regression_examples.md"
md.write_text("\n".join(lines), encoding="utf-8")
print(f"\nBase = {payload['baseline_score']:.4f}; FT = {payload['ft_score']:.4f}; THUA = {len(losses)}")
for r in selected:
    print(json.dumps(r, ensure_ascii=False, indent=2))
print("Saved:", out, md)
