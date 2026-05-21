#!/bin/sh
# Fix only dummy (Your Name) identity; strip Cursor trailers; leave other contributors unchanged.
set -e

MSG=$(git log -1 --format=%B | grep -v '^Co-authored-by:' | grep -v '^Made-with:' || true)
AUTHOR=$(git log -1 --format=%an)
EMAIL=$(git log -1 --format=%ae)
HAS_TRAILERS=$(git log -1 --format=%B | grep -cE '^(Co-authored-by:|Made-with:)' || true)

needs_identity=0
if [ "$AUTHOR" = "Your Name" ] || [ "$EMAIL" = "your@email.com" ]; then
  needs_identity=1
fi

if [ "$needs_identity" -eq 0 ] && [ "$HAS_TRAILERS" -eq 0 ]; then
  exit 0
fi

printf '%s\n' "$MSG" > /tmp/git-fix-commit-msg.txt

if [ "$needs_identity" -eq 1 ]; then
  git commit --amend --allow-empty --author="ussoewwin <136552381+ussoewwin@users.noreply.github.com>" -F /tmp/git-fix-commit-msg.txt
else
  git commit --amend --allow-empty -F /tmp/git-fix-commit-msg.txt
fi
