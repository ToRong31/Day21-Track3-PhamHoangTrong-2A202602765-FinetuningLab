"""Build rubric Option B with local verification sources and bonus evidence."""
import datetime
import hashlib
import json
import pathlib
import shutil
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = "lab21_2A202602765"
OUTPUT = ROOT / "submission" / NAME
ZIP = ROOT / "submission" / (NAME + ".zip")
IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "*.pyo", "*.crdownload")


def checked_move(source, destination):
    for path in (source, destination):
        assert path.resolve().is_relative_to(ROOT), "Archive path is outside workspace"
    assert not destination.exists(), "Archive destination already exists"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    history = ROOT / "backups/submissions" / stamp
    if OUTPUT.exists():
        checked_move(OUTPUT, history / NAME)
    if ZIP.exists():
        checked_move(ZIP, history / ZIP.name)
    OUTPUT.mkdir(parents=True)
    for name in ["src", "notebooks", "scripts", "tests", "data", "docs", "colab", "solutions"]:
        shutil.copytree(ROOT / name, OUTPUT / name, ignore=IGNORE)
    shutil.copytree(ROOT / "results", OUTPUT / "results", ignore=IGNORE)
    for name in ["README.md", "rubric.md", "BONUS-CHALLENGE.md", "BONUS-CHALLENGE-EN.md",
                 "HARDWARE-GUIDE.md", "SIMULATION-FINDINGS.md", "VIBE-CODING.md", "LINKS.md",
                 "LICENSE", "Makefile", "pyproject.toml", "requirements.txt", "requirements-cpu.txt", ".env.example"]:
        shutil.copy2(ROOT / name, OUTPUT / name)
    (OUTPUT / "submission").mkdir()
    for name in ["REPORT.md", "REFLECTION.md"]:
        shutil.copy2(ROOT / "submission" / name, OUTPUT / "submission" / name)
    b2 = OUTPUT / "bonus/B2_UIT"
    for name in ["data", "results"]:
        shutil.copytree(ROOT / "bonus/B2_UIT" / name, b2 / name, ignore=IGNORE)
    # B2 changes its prompt configuration; retain the shared library plus that override.
    shutil.copytree(ROOT / "src", b2 / "src", ignore=IGNORE)
    shutil.copy2(ROOT / "bonus/B2_UIT/src/labkit/config.py", b2 / "src/labkit/config.py")
    shutil.copytree(ROOT / "bonus/B4", OUTPUT / "bonus/B4", ignore=IGNORE)
    readme = """# Lab21 submission: Option B

Start with submission/REPORT.md. LINKS.md contains the GitHub source URL and the
public main adapter on HuggingFace Hub. results/ contains measured core and bonus
summaries; bonus/B2_UIT and bonus/B4 retain separate evidence.

Source, notebooks, tests and frozen data are included for verification. No model
weights or historical backup ZIPs are included: the core adapter is public on Hub.
Run python scripts/verify.py and python scripts/verify_b4.py from this folder.
B2 review is by AI; B3 is incomplete. A FAILED model verdict is a reported result.
MANIFEST_SHA256.json records the bytes of every submitted file except itself.
"""
    (OUTPUT / "README_SUBMISSION.md").write_text(readme, encoding="utf-8")
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
    for path in files:
        assert path.suffix not in {".safetensors", ".bin", ".token", ".pyc", ".crdownload"}
        assert path.name not in {".env", ".env.local"}
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    hashes = {path.relative_to(OUTPUT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    manifest = OUTPUT / "MANIFEST_SHA256.json"
    manifest.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files + [manifest]:
            archive.write(path, NAME + "/" + path.relative_to(OUTPUT).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(NAME + "/" + name)).hexdigest() == expected
    print(f"Created {ZIP.name}: {ZIP.stat().st_size:,} bytes; {len(hashes)} files verified.")


if __name__ == "__main__":
    main()
