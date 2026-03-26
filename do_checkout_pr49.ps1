# Check out PR #49 locally
# Run: powershell -ExecutionPolicy Bypass -File do_checkout_pr49.ps1

Set-Location "d:\USERFILES\GitHub\ComfyUI-QwenImageLoraLoader"

# Remove stale index lock if present
if (Test-Path ".git\index.lock") {
    Remove-Item ".git\index.lock" -Force
}

# Check out branch pr-49
git checkout pr-49
