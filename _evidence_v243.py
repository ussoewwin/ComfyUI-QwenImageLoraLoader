# -*- coding: utf-8 -*-
import pathlib
import hashlib

root = pathlib.Path(r"D:\USERFILES\GitHub\ComfyUI-QwenImageLoraLoader")

def digest(p: pathlib.Path) -> str:
    b = p.read_bytes()
    return hashlib.sha256(b).hexdigest()[:16]

def stats(p: pathlib.Path) -> dict:
    t = p.read_text(encoding="utf-8")
    return {
        "path": str(p.relative_to(root)).replace("\\", "/"),
        "bytes": len(t.encode("utf-8")),
        "chars": len(t),
        "lines": t.count("\n") + 1,
        "sha256_prefix": digest(p),
    }

short = root / "release" / "_release_v2.4.3_body_en.md"
full_en = root / "zhmd" / "v2.4.3.md"
rel_notes = root / "RELEASE_NOTES" / "v2.4.3.md"

rows = [stats(short), stats(full_en), stats(rel_notes)]
print("=== FILE STATS (evidence) ===")
for r in rows:
    print(r)

print("\n=== BYTE-IDENTICAL? ===")
print("short vs zhmd/v2.4.3.md:", short.read_text(encoding="utf-8") == full_en.read_text(encoding="utf-8"))
print("short vs RELEASE_NOTES/v2.4.3.md:", short.read_text(encoding="utf-8") == rel_notes.read_text(encoding="utf-8"))

print("\n=== HEADINGS in zhmd NOT in short release body ===")
import re
h = re.findall(r"^## .+", full_en.read_text(encoding="utf-8"), re.MULTILINE)
short_h = set(re.findall(r"^## .+", short.read_text(encoding="utf-8"), re.MULTILINE))
missing = [x for x in h if x not in short_h]
print("count missing in release/_release_v2.4.3_body_en.md:", len(missing))
for m in missing[:15]:
    print(" ", m)
if len(missing) > 15:
    print(" ...", len(missing) - 15, "more")

print("\n=== git 67b3463: both files added same size ===")
import subprocess
out = subprocess.check_output(
    ["git", "show", "67b3463", "--stat", "--", "release/_release_v2.4.3_body_en.md", "RELEASE_NOTES/v2.4.3.md"],
    cwd=root,
    text=True,
    encoding="utf-8",
    errors="replace",
)
print(out.strip())
