# Git Authentication Fix Procedure (When Broken in Another Chat)

## Actions Taken

1. **Removed GitHub authentication from Windows Credential Manager**
   - `git:https://github.com`
   - `gh:github.com:ussoewwin`
   - `gh:github.com:`
   - → Resolved issues caused by invalid/old tokens being cached

2. **Backed up `.git-credentials`**
   - Renamed `%USERPROFILE%\.git-credentials` to `%USERPROFILE%\.git-credentials.bak`
   - Prevented use of old tokens that remained in the file

## What You Need to Do

### 1. Clone / Push Using an Interactive Terminal

**Important**: Execute `git clone` or `git push` in an **interactive terminal** (PowerShell, Git Bash, VS Code terminal, etc.).

- Since the cache has been cleared, the **Git Credential Manager (GCM) login prompt** will appear next time.
- Sign in to GitHub on that screen (browser or token input), and the credentials will be saved again, allowing clone/push to work thereafter.
- Running `git` in background execution or environments without prompts will fail because authentication input cannot be provided.

### 2. Revoke Old Personal Access Token (PAT)

**Security**: The GitHub PAT that was previously saved in `%USERPROFILE%\.git-credentials` may have been exposed in logs, etc.

1. GitHub → **Settings** → **Developer settings** → **Personal access tokens**
2. **Revoke** the relevant token (or any old tokens you don't remember) to invalidate it
3. If necessary, **Generate new token** to create a new PAT, and enter **that new token** in the GCM prompt to use it

### 3. Current Git Configuration (Reference)

- `credential.helper=manager` (using Git Credential Manager)
- `.git-credentials` has been backed up, so it won't be used unless `credential.helper=store` is set

## When It "Stops" or "Hangs"

- **No response after `Cloning into...`** → The GCM login screen may be in the background. Bring the **"Git Credential Manager"** or **browser** window to the foreground from the taskbar and operate it.
- **Terminal appears frozen** → Often waiting for authentication. Operating the above window will resume it.
- **Nothing appears / keeps stopping** → Try reinstalling Git. If it still persists, temporarily remove `credential.helper` and clone using **username + PAT in the URL** method (`git clone https://USERNAME:TOKEN@github.com/...`). Create a new PAT on GitHub beforehand.

## If Clone Still Doesn't Work

- Verify that the above is executed in **an interactive terminal**
- Check that `git config --global credential.helper` is still `manager`
- Check if `github.com` is blocked by firewall / VPN / proxy
- If an error message appears, note the full text and present it to support for easier cause identification

## Recovery Examples

- **Git reinstallation** has successfully restored `git clone` (after removing Credential Manager and backing up `.git-credentials`, reinstallation → clone worked in an interactive terminal).
- As the most reliable and quickest solution, **trying reinstallation first** is also an option.
