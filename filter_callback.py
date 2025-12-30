import re

def callback(blob, metadata):
    if metadata.path != b'README.md':
        return blob
    
    try:
        if isinstance(blob, bytes):
            content = blob.decode('utf-8')
        else:
            content = blob
        
        lines = content.split('\n')
        pattern = re.compile(r'[あ-ん]|[ア-ン]|[一-龯]')
        filtered_lines = [line for line in lines if not pattern.search(line)]
        filtered_content = '\n'.join(filtered_lines)
        
        if isinstance(blob, bytes):
            return filtered_content.encode('utf-8')
        else:
            return filtered_content
    except Exception:
        return blob

