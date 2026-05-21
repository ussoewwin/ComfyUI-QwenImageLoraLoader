# -*- coding: utf-8 -*-
import hashlib
import pathlib
import subprocess

root = pathlib.Path(r"D:\USERFILES\GitHub\ComfyUI-QwenImageLoraLoader")

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def stats_text(p: pathlib.Path) -> dict:
    t = p.read_text(encoding="utf-8")
    b = t.encode("utf-8")
    return {
        "path": str(p.relative_to(root)).replace("\\", "/"),
        "lines": t.count("\n") + 1,
        "utf8_bytes": len(b),
        "sha256": sha256_bytes(b),
    }

def main():
    short = root / "release" / "_release_v2.4.3_body_en.md"
    full_en = root / "zhmd" / "v2.4.3.md"

    rows = [stats_text(short), stats_text(full_en)]

    parent_blob = subprocess.check_output(
        ["git", "rev-parse", "8e1e9d6^:release/_release_v2.4.3_body_en.md"],
        cwd=root,
        text=True,
    ).strip()
    head_blob = subprocess.check_output(
        ["git", "rev-parse", "HEAD:release/_release_v2.4.3_body_en.md"],
        cwd=root,
        text=True,
    ).strip()

    print("=== ENGLISH-ONLY EVIDENCE (not Chinese comparison) ===\n")
    for r in rows:
        print(r)

    print("\n=== GITHUB RELEASE BODY: parent-of-shorten vs HEAD (byte identity) ===")
    print("parent_blob (8e1e9d6^):", parent_blob)
    print("HEAD blob:            ", head_blob)
    print("IDENTICAL since 8e1e9d6:", parent_blob == head_blob)

    parent_path = root / "release" / "_release_v2.4.3_body_en.md"
    parent_export = subprocess.check_output(
        ["git", "show", f"{parent_blob}:{parent_path}"],
        cwd=root,
    )
    head_export = short.read_text(encoding="utf-8")

    print("\n=== LINE COUNT (English release/_release_v2.4.3_body_en.md only) ===")
    print("parent (before agent -650 lines in 8e1e9d6):", stats_text(parent_path)["lines"], "lines")
    print("HEAD (current file):              ", stats_text(short)["lines"], "lines")

    print("\n=== SHA256 (English release/_release_v2.4.3_body_en.md only) ===")
    print("parent file:", sha256_bytes(parent_export.encode("utf-8")))
    print("HEAD file:  ", sha256_bytes(head_export.encode("utf-8")))
    print("BYTE-IDENTICAL parent export vs HEAD file:", parent_export == head_export)

    print("\n=== FULL ENGLISH TECH NOTE (zhmd) — NOT the same as release body (contrast only, not proof) ===")
    print("zhmd/v2.4.3.md lines:", stats_text(full_en)["lines"])
    print("release/_release_v2.4.3_body_en.md lines:", stats_text(short)["lines"])
    print("sha256 zhmd != release body:", stats_text(full_en)["sha256"] != stats_text(short)["sha256"])

    print("\n=== git show 8e1e9d6 (who shortened release body) ===")
    stat = subprocess.check_output(
        ["git", "show", "8e1e9d6", "--stat", "--", "release/_release_v2.4.3_body_en.md"],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(stat.strip())

if __name__ == "__main__":
    main()
