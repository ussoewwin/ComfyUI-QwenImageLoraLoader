# PR #49 merge on GitHub (Squash and merge)
# Disable proxy to avoid 127.0.0.1 connection

$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:http_proxy = ""
$env:https_proxy = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

Set-Location $PSScriptRoot
gh pr merge 49 --squash
