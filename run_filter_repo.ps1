$code = Get-Content filter_callback.py -Raw
git filter-repo --path README.md --blob-callback $code --force

