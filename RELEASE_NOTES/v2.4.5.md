<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.4.5.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

Documentation release: full Chinese README and bilingual language switchers on the English and Chinese README and release-note pages. No changes to node code or runtime behavior.

---

## What's new

### Chinese documentation (`zhmd/`)

- **`zhmd/README.md`**: Chinese translation of the repository README (installation, features, node screenshots, changelog, and links).
- **`zhmd/v2.4.4.md`**: Chinese release notes for v2.4.4 (paired with the English v2.4.4 release page).
- Future versioned release notes for this series live under `zhmd/` (e.g. this file: `zhmd/v2.4.5.md`).

### Bilingual language switchers

- Root **`README.md`**: EN ↔ 中文 switcher (HTML table + PNG flag buttons) linking to `zhmd/README.md`.
- **`zhmd/README.md`**: 中文 ↔ EN switcher linking back to the English README.
- **Release notes** (GitHub Releases and repo copies): same switcher pattern on English (`RELEASE_NOTES/v*.md`) and Chinese (`zhmd/v*.md`) pages.

### GitHub rendering fix

- **`zhmd/README.md`**: Node screenshot paths use `../images/` so images resolve to the repo-root `images/` folder on GitHub (fixes broken `images/` paths that pointed at `zhmd/images/`).

---

## Repository layout (docs)

| English | Chinese |
|---------|---------|
| `README.md` | `zhmd/README.md` |
| `RELEASE_NOTES/v2.4.x.md` | `zhmd/v2.4.x.md` |
| GitHub Release body (EN) | Linked from switcher → `zhmd/v2.4.x.md` on `main` |

Changelog bullets in both READMEs are kept in sync under **Changelog** / **更新日志**.

---

## Upgrade notes

- **No install or code update required** for this release if you only use the nodes; pull or refresh docs if you want the Chinese README.
- Open [中文 README](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/README.md) from the switcher on the main README.

---

## Commits in this release (since v2.4.4)

| Commit | Summary |
|--------|---------|
| `04661d1` | Add Chinese README and language switcher links |
| `d617c63`–`9b9cfe8` | README language switcher UI (alignment, PNG assets, HTML table) |
| `b5de9ca` | Move Chinese docs to `zhmd/`, fix bilingual links |
| `558e1c3` | Release note source paths under `release/` |
| `646f5d2` | Changelog entries for v2.4.5 (EN + ZH README) |
| `b7f2805` | Fix Chinese README image paths (`../images/`) |
