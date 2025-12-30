$env:FILTER_BRANCH_SQUELCH_WARNING = "1"
$repoRoot = (Get-Location).Path
git filter-branch --force --index-filter "git rm -f --cached --ignore-unmatch README.md && git checkout HEAD -- README.md 2>nul && git show HEAD:remove_japanese.py > remove_japanese.py 2>nul && python remove_japanese.py README.md && rm -f remove_japanese.py && git add README.md" --prune-empty --tag-name-filter cat -- --all

