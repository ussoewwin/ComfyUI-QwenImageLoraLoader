#!/bin/sh
if [ "$GIT_AUTHOR_NAME" = "Your Name" ] || [ "$GIT_AUTHOR_EMAIL" = "your@email.com" ]; then
  export GIT_AUTHOR_NAME=ussoewwin
  export GIT_AUTHOR_EMAIL=136552381+ussoewwin@users.noreply.github.com
fi
if [ "$GIT_COMMITTER_NAME" = "Your Name" ] || [ "$GIT_COMMITTER_EMAIL" = "your@email.com" ]; then
  export GIT_COMMITTER_NAME=ussoewwin
  export GIT_COMMITTER_EMAIL=136552381+ussoewwin@users.noreply.github.com
fi
