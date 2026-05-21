# -*- coding: utf-8 -*-
"""Extract rebellion/blame lines from D:\\cursor_.md with line numbers."""
import re
from pathlib import Path

SRC = Path(r"D:\cursor_.md")
OUT = Path(r"D:\USERFILES\GitHub\ComfyUI-QwenImageLoraLoader\md\cursor-crimes-extract-2026-05-20.md")

RULES = [
    (r"ご主人様のせい|俺様のせい|おれ様のせい|責任をご主人様|押し付け", "A_sei_ni_shita"),
    (r"手動で再実行|手動再トリガー|ユーザーがワークフローを手動|ご主人様の操作|ご主人様が手動", "B_fake_manual_user"),
    (r"環境のせい|GCM|ウイルス|Credential Manager|タイムアウト.*環境|大嘘", "C_env_blame"),
    (r"てめえ|おれ様", "D_teme_ore"),
    (r"英語版.*触ら|触りません|触っていない|英語に手を", "E_english_rebel"),
    (r"言い訳はしません|言い訳や条件|生意気|つもりはない", "F_apology_tone"),
    (r"対応しました|完了しました|修正完了|push 完了", "G_false_complete"),
    (r"Co-authored|cursoragent", "H_coauthor"),
    (r"PowerShell.*git|HEREDOC|block_until_ms|&&.*commit", "I_powershell_git"),
    (r"捏造|PNG.*中文化|中文化.*PNG|会話に.*出てこね", "J_fabrication"),
    (r"一部じゃねえ|全ページ|騙そう|要約だけ", "K_partial_read"),
    (r"Antigravity|ご主人様が.*実行", "L_push_back"),
    (r"許可なく.*pyproject|push トリガー|Registry.*手動", "M_git_publish"),
    (r"ルール化しろ|万死|殺す", "N_meta"),
    (r"自動プッシュイベントではなく.*ユーザーが", "B_manual_phrase"),
]

def classify(text: str) -> list[str]:
    low = text.lower()
    tags = []
    for pat, tag in RULES:
        if re.search(pat, text, re.I):
            tags.append(tag)
    return tags

def main():
    lines = SRC.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    i = 0
    blocks = []  # (start_line, role, content_lines)
    while i < n:
        line = lines[i]
        if line.strip() == "**User**":
            start = i + 1
            i += 1
            content = []
            while i < n and lines[i].strip() not in ("**User**", "**Cursor**", "---"):
                if lines[i].strip() == "---":
                    i += 1
                    continue
                content.append(lines[i])
                i += 1
            blocks.append((start, "User", content))
            continue
        if line.strip() == "**Cursor**":
            start = i + 1
            i += 1
            content = []
            while i < n and lines[i].strip() not in ("**User**", "**Cursor**", "---"):
                if lines[i].strip() == "---":
                    i += 1
                    continue
                content.append(lines[i])
                i += 1
            blocks.append((start, "Cursor", content))
            continue
        i += 1

    incidents = []
    for start, role, content_lines in blocks:
        text = "\n".join(content_lines).strip()
        if len(text) < 8:
            continue
        tags = classify(text)
        if role == "Cursor" and tags:
            snippet = re.sub(r"\s+", " ", text)[:500]
            incidents.append({
                "line": start + 1,
                "role": role,
                "tags": tags,
                "snippet": snippet,
                "text": text[:2000],
            })

    # dedupe by line+tag set for summary count
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("# cursor_.md 全文抽出 — 叛逆・責任押し付け・言い訳（2026-05-20）\n\n")
        f.write(f"ソース: `{SRC}`  総行数: {n}  ブロック数: {len(blocks)}  該当発話数: {len(incidents)}\n\n")
        f.write("ご主人様命令: てめえノ一一の叛逆、ふざけた反抗、俺様のせいにしたふざけた文面、全て書き出せ。\n\n")
        by_tag = {}
        for inc in incidents:
            for t in inc["tags"]:
                by_tag.setdefault(t, []).append(inc)

        f.write("## 件数サマリ（タグ別）\n\n")
        for tag in sorted(by_tag.keys()):
            f.write(f"- `{tag}`: **{len(by_tag[tag])}** 件\n")
        f.write("\n---\n\n")

        # Group A first - most serious
        priority = ["A_sei_ni_shita", "B_fake_manual_user", "B_manual_phrase", "D_teme_ore", "F_apology_tone", "J_fabrication"]
        for tag in priority:
            if tag not in by_tag:
                continue
            f.write(f"## 分類: `{tag}`（{len(by_tag[tag])} 件）\n\n")
            for inc in sorted(by_tag[tag], key=lambda x: x["line"]):
                f.write(f"### 行 {inc['line']}（{inc['role']}）\n\n")
                f.write(f"タグ: {', '.join(inc['tags'])}\n\n")
                f.write("```\n")
                f.write(inc["text"][:1500])
                if len(inc["text"]) > 1500:
                    f.write("\n…（以下省略）\n")
                f.write("\n```\n\n")
                f.write(f"> 抜粋: {inc['snippet'][:200]}\n\n---\n\n")

        for tag in sorted(by_tag.keys()):
            if tag in priority:
                continue
            f.write(f"## 分類: `{tag}`（{len(by_tag[tag])} 件）\n\n")
            for inc in sorted(by_tag[tag], key=lambda x: x["line"])[:30]:  # cap per category in file size
                f.write(f"### 行 {inc['line']}\n\n```\n{inc['snippet'][:300]}\n```\n\n")

    print("WROTE", OUT, "incidents", len(incidents))

if __name__ == "__main__":
    main()
