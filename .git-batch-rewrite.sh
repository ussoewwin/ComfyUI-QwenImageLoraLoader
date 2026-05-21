#!/bin/sh
# Rewrite history in batches: only Your Name / your@email.com; strip Cursor trailers.
# Leaves avan, avan06, and other contributors unchanged.
set -e
export FILTER_BRANCH_SQUELCH_WARNING=1
REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$REPO_ROOT"

BATCH=50
TOTAL=$(git rev-list --count main)
END=$BATCH

while [ "$END" -le "$TOTAL" ]; do
  TIP=$(git rev-list --reverse main | sed -n "${END}p")
  echo "=== Batch end=$END / $TOTAL tip=$TIP ==="
  git filter-branch -f \
    --env-filter "sh \"$REPO_ROOT/.git-env-filter.sh\"" \
    --msg-filter "grep -v '^Co-authored-by:' | grep -v '^Made-with:' || true" \
    "$TIP"
  rm -rf .git/refs/original
  END=$((END + BATCH))
done

echo "=== Final pass (main) ==="
git filter-branch -f \
  --env-filter "sh \"$REPO_ROOT/.git-env-filter.sh\"" \
  --msg-filter "grep -v '^Co-authored-by:' | grep -v '^Made-with:' || true" \
  main
rm -rf .git/refs/original

echo "=== Done. Author summary ==="
git shortlog -sne main | head -20

echo "=== avan commits (must stay avan) ==="
git log main --format="%h %an <%ae> %s" --author=avan | head -5
