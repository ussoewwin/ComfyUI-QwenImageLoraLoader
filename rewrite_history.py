#!/usr/bin/env python3
"""
Rewrite git history to remove Japanese characters from README.md in all commits.
"""
import re
import subprocess
import sys
import os
import tempfile

def remove_japanese_from_content(content):
    """Remove lines containing Japanese characters from content."""
    japanese_pattern = re.compile(r'[あ-ん]|[ア-ン]|[一-龯]')
    lines = content.split('\n')
    filtered_lines = [line for line in lines if not japanese_pattern.search(line)]
    return '\n'.join(filtered_lines)

def get_all_commits():
    """Get all commit hashes."""
    result = subprocess.run(
        ['git', 'log', '--all', '--format=%H'],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    return result.stdout.strip().split('\n') if result.stdout.strip() else []

def process_commit(commit_hash):
    """Process a single commit to remove Japanese from README.md."""
    # Check if README.md exists in this commit
    result = subprocess.run(
        ['git', 'cat-file', '-e', f'{commit_hash}:README.md'],
        capture_output=True,
        cwd=os.getcwd()
    )
    if result.returncode != 0:
        return False  # README.md doesn't exist in this commit
    
    # Get README.md content
    result = subprocess.run(
        ['git', 'show', f'{commit_hash}:README.md'],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    if result.returncode != 0:
        return False
    
    original_content = result.stdout
    filtered_content = remove_japanese_from_content(original_content)
    
    # If content changed, we need to rewrite this commit
    if original_content != filtered_content:
        return True
    return False

def main():
    print("Getting all commits...")
    commits = get_all_commits()
    print(f"Found {len(commits)} commits")
    
    # Use git filter-branch with a custom script
    filter_script = '''#!/bin/bash
git rm --cached --ignore-unmatch README.md
git checkout HEAD -- README.md 2>/dev/null || true
if [ -f README.md ]; then
    python remove_japanese.py README.md
    git add README.md
fi
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(filter_script)
        script_path = f.name
    
    try:
        os.chmod(script_path, 0o755)
        env = os.environ.copy()
        env['FILTER_BRANCH_SQUELCH_WARNING'] = '1'
        
        result = subprocess.run(
            ['git', 'filter-branch', '--force', '--index-filter', f'bash {script_path}',
             '--prune-empty', '--tag-name-filter', 'cat', '--', '--all'],
            cwd=os.getcwd(),
            env=env
        )
        
        if result.returncode == 0:
            print("Successfully rewrote git history")
        else:
            print(f"Error: git filter-branch returned {result.returncode}")
            sys.exit(1)
    finally:
        os.unlink(script_path)

if __name__ == '__main__':
    main()

