"""Retry only B1 adapter switching in a fresh process, without repeating merge."""
import json
import pathlib
import sys

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from labkit import generate
from labkit.config import get_tier
from peft import PeftModel

merge_path = ROOT / "results/merge_check.json"
assert merge_path.exists(), "Missing merge proof; complete the merge check first."
merge = json.loads(merge_path.read_text(encoding="utf-8"))
assert merge["delta"] >= -merge["tolerance"]
for name in ("correct", "attn_only"):
    assert (ROOT / "adapters" / name / "adapter_model.safetensors").exists()

model, tok = generate.load_base(get_tier("T4"))
model = PeftModel.from_pretrained(model, str(ROOT / "adapters/correct"), adapter_name="correct")
model.load_adapter(str(ROOT / "adapters/attn_only"), adapter_name="attn_only")
model.eval()
sample = json.loads((ROOT / "data/eval_target.jsonl").read_text(encoding="utf-8").splitlines()[0])
owner = id(model)
rows = []
for name in ("correct", "attn_only"):
    model.set_adapter(name)
    assert id(model) == owner
    pred, _ = generate.generate_batch(model, tok, [sample["input"]],
                                     system=generate.NAIVE_PROMPT, label=name)
    rows.append({"adapter": name, "prediction": pred[0]})
    print(f"[{name}] {pred[0]}")
proof = {"same_loaded_model": True, "adapters": [r["adapter"] for r in rows],
         "ticket": sample["input"], "outputs": rows,
         "note": "Sequential adapter selection with set_adapter on one loaded base; not concurrent serving."}
out = ROOT / "results/hotswap_check.json"
out.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
print("B1 adapter switching complete:", out)
del model
generate.free_memory()
