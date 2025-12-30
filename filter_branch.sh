#!/bin/bash
export FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch --force --index-filter '
    git rm --cached --ignore-unmatch README.md
    git checkout HEAD -- README.md 2>/dev/null || true
    python remove_japanese.py README.md
    git add README.md
' --prune-empty --tag-name-filter cat -- --all

