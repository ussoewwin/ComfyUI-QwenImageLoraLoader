# Temporary commit-callback for git filter-repo (deleted after rewrite)
if commit.author_name == b"Your Name" or commit.author_email == b"your@email.com":
    commit.author_name = b"ussoewwin"
    commit.author_email = b"136552381+ussoewwin@users.noreply.github.com"
if commit.committer_name == b"Your Name" or commit.committer_email == b"your@email.com":
    commit.committer_name = b"ussoewwin"
    commit.committer_email = b"136552381+ussoewwin@users.noreply.github.com"
lines = commit.message.split(b"\n")
lines = [
    line
    for line in lines
    if not (line.startswith(b"Co-authored-by:") or line.startswith(b"Made-with:"))
]
commit.message = b"\n".join(lines)
if commit.message and not commit.message.endswith(b"\n"):
    commit.message += b"\n"
