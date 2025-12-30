#!/usr/bin/env python3
"""
Filter function for git-filter-repo to remove Japanese from README.md
"""
import re

def callback(blob, metadata):
    """Remove lines containing Japanese characters from README.md"""
    if metadata.path != b'README.md':
        return blob
    
    try:
        if isinstance(blob, bytes):
            content = blob.decode('utf-8')
        else:
            content = blob
        
        lines = content.split('\n')
        
        # Pattern to match Japanese characters (hiragana, katakana, kanji)
        japanese_pattern = re.compile(r'[あ-ん]|[ア-ン]|[一-龯]')
        
        filtered_lines = [line for line in lines if not japanese_pattern.search(line)]
        filtered_content = '\n'.join(filtered_lines)
        
        if isinstance(blob, bytes):
            return filtered_content.encode('utf-8')
        else:
            return filtered_content
    except Exception:
        # If decoding fails, return original blob
        return blob

