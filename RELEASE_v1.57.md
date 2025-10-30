# Release v1.57: Duplicate Installation Prevention

**Release Date**: January 2025  
**Reporter**: [@ussoewwin](https://github.com/ussoewwin)

## Summary

Fixed critical bug where running the installer multiple times would create duplicate integration blocks in `ComfyUI-nunchaku/__init__.py`, breaking the plugin. This release implements comprehensive duplicate prevention mechanisms.

## The Problem

Running the installer multiple times (e.g., during troubleshooting, updates, or accidental double-clicks) would repeatedly append the same integration code to `__init__.py`, creating:
- Multiple identical blocks
- Conflicts in node registration
- Broken plugin functionality
- Manual cleanup required

Additionally, the backup file would be overwritten on second run, making recovery impossible.

## Modified Files

The following files were modified to implement duplicate prevention and backup consolidation:

1. **`append_integration.py`** - Core integration appending logic with marker-based duplicate detection
2. **`remove_integration.py`** - Block removal for both new and legacy formats (NEW FILE)
3. **`install_qwen_lora.bat`** - Global Python installer (removed duplicate backup creation)
4. **`install_qwen_lora_portable.bat`** - Portable/embedded Python installer (removed duplicate backup creation)
5. **`uninstall_qwen_lora.bat`** - Global Python uninstaller (unified to `.qwen_image_backup` only)
6. **`uninstall_qwen_lora_portable.bat`** - Portable/embedded Python uninstaller (NEW FILE, unified to `.qwen_image_backup` only)
7. **`README.md`** - Added v1.57 changelog entry with release link

---

### 1. `append_integration.py`

#### Marker-Based Duplicate Detection

**Lines 49-51**: Implements duplicate prevention by checking for existing installation markers

```python
if BEGIN_MARKER in content and END_MARKER in content:
    print("Integration code already present. Skipping.")
    return True
```

- **Line 49**: Checks if both BEGIN_MARKER and END_MARKER exist in the file
- **Line 50**: Prints informative message to user
- **Line 51**: Returns True to exit gracefully without modifications
- **Result**: If integration block is already present, the script skips installation entirely

#### Backup Protection

**Lines 54-57**: Ensures backup file is only created once

```python
backup_path = init_py_path + ".qwen_image_backup"
if not os.path.exists(backup_path):
    shutil.copy2(init_py_path, backup_path)
    print(f"Backup created: {backup_path}")
```

- **Line 54**: Defines unique backup file path with `.qwen_image_backup` extension
- **Line 55**: Checks if backup file already exists
- **Line 56**: Creates backup only if it doesn't exist yet
- **Line 57**: Prints confirmation message
- **Result**: No overwriting of existing backups, allowing safe reinstallation

### 2. `append_integration.py` (Revised)

#### Corrected Backup Creation Order

**Lines 45-75**: Fixed backup creation to happen BEFORE append, but AFTER duplicate check

```python
try:
    # Read file content
    with open(init_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already installed
    if BEGIN_MARKER in content and END_MARKER in content:
        print("Integration code already present. Skipping.")
        return True
    
    # Create backup only if not exists and no integration code
    backup_path = init_py_path + ".qwen_image_backup"
    if not os.path.exists(backup_path):
        try:
            shutil.copy2(init_py_path, backup_path)
            print(f"Backup created: {backup_path}")
        except Exception as backup_error:
            print(f"Warning: Could not create backup: {backup_error}")
            # Continue anyway - the block removal function can handle this
    
    # Append integration code
    with open(init_py_path, 'a', encoding='utf-8') as f:
        if not content.endswith('\n'):
            f.write('\n')
        f.write(integration_code)
    
    print(f"Successfully appended integration code to: {init_py_path}")
    return True
except Exception as e:
    print(f"Error appending integration code: {e}")
    return False
```

**Key Changes**:
- **Lines 46-48**: Read content first into variable scope
- **Lines 50-53**: Check for duplicate BEFORE creating backup
- **Lines 55-63**: Create backup ONLY if no duplicates AND backup doesn't exist
- **Result**: Backup file always contains original state (before any integration blocks)

### 3. `remove_integration.py`

#### Legacy Block Removal

**Lines 39-67**: Removes old-style blocks without markers

```python
def _remove_legacy_block(text: str) -> tuple[str, int]:
    """Remove old-style integration block without markers."""
    old_comment = "# ComfyUI-QwenImageLoraLoader Integration"
    removed = 0
    
    start = text.find(old_comment)
    if start == -1:
        return text, 0
    
    # Find the try-except block that starts after the comment
    pattern = r'# ComfyUI-QwenImageLoraLoader Integration\ntry:.*?except ImportError:.*?logger\.exception\(.*?\)'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        start_pos = match.start()
        end_pos = match.end()
        
        # Trim leading whitespace from the block
        while start_pos > 0 and text[start_pos - 1] in "\r\n\t ":
            start_pos -= 1
        
        # Remove the block
        text = text[:start_pos].rstrip() + "\n\n" + text[end_pos:].lstrip()
        removed = 1
    
    return text, removed
```

- **Line 41**: Defines old-style comment marker
- **Line 45**: Searches for old comment pattern
- **Line 51**: Uses regex to match entire try-except block
- **Lines 56-57**: Calculates block boundaries
- **Lines 59-61**: Trims leading whitespace
- **Line 64**: Removes block cleanly
- **Result**: Old integration blocks removed even without markers

#### Complete Block Removal Logic

**Lines 70-89**: Removes both new and legacy blocks

```python
def _remove_all_blocks(text: str) -> tuple[str, int]:
    # First, remove new-style blocks with markers
    removed = 0
    start = text.find(BEGIN_MARKER)
    while start != -1:
        stop = text.find(END_MARKER, start)
        if stop == -1:
            break
        stop += len(END_MARKER)
        while stop < len(text) and text[stop] in "\r\n":
            stop += 1
        text = (text[:start].rstrip() + "\n\n" + text[stop:].lstrip())
        removed += 1
        start = text.find(BEGIN_MARKER)
    
    # Then, remove old-style blocks without markers
    text, legacy_removed = _remove_legacy_block(text)
    removed += legacy_removed
    
    return text, removed
```

- **Lines 71-83**: Removes new-style blocks with BEGIN/END markers
- **Line 86**: Calls legacy removal function
- **Line 87**: Combines removal counts
- **Result**: All integration blocks removed regardless of format

### 3. `install_qwen_lora.bat` and `install_qwen_lora_portable.bat`

#### Removed Duplicate Backup Creation

**Lines 48-54 (Global) / Lines 75-81 (Portable)**: Simplified to remove redundant backup logic

**Global Installer (`install_qwen_lora.bat`)**:
```bat
REM Check if already installed
findstr /C:"ComfyUI-QwenImageLoraLoader Integration" "%NUNCHAKU_PATH%\__init__.py" >nul 2>&1
if %errorlevel% equ 0 (
    echo Already installed. Integration code already exists in __init__.py
    pause
    exit /b 0
)

echo Adding LoRA loader integration code...

REM Append integration code using Python script
py -3 "%LORA_LOADER_PATH%\append_integration.py" "%NUNCHAKU_PATH%\__init__.py"
```

**Portable Installer (`install_qwen_lora_portable.bat`)**:
```bat
REM Check if already installed
findstr /C:"ComfyUI-QwenImageLoraLoader Integration" "%NUNCHAKU_PATH%\__init__.py" >nul 2>&1
if %errorlevel% equ 0 (
    echo Already installed. Integration code already exists in __init__.py
    pause
    exit /b 0
)

echo Adding LoRA loader integration code...

REM Append integration code using embedded Python
"%PYTHON_CMD%" "%LORA_LOADER_PATH%\append_integration.py" "%NUNCHAKU_PATH%\__init__.py"
```

**Key Changes**:
- **Before**: Batch files created `.backup` file, then `append_integration.py` created `.qwen_image_backup`
- **After**: Only `append_integration.py` creates backup (`.qwen_image_backup` only)
- **Benefit**: Prevents backup from being created after integration block is already added
- **Both installers**: Added duplicate check using `findstr` before calling Python script

### 4. `uninstall_qwen_lora.bat` and `uninstall_qwen_lora_portable.bat`

#### Uninstallation Strategy

**Lines 23-43 (Global) / Lines 47-65 (Portable)**: Implements two-stage uninstallation approach

**Global Uninstaller (`uninstall_qwen_lora.bat`)**:
```bat
echo Checking for backups...

REM Try new-style backup first
if exist "%NUNCHAKU_PATH%\__init__.py.qwen_image_backup" (
    copy "%NUNCHAKU_PATH%\__init__.py.qwen_image_backup" "%NUNCHAKU_PATH%\__init__.py" >nul
    echo Restored from backup: __init__.py.qwen_image_backup
    echo [OK] Qwen Image LoRA Loader integration removed.
    pause
    exit /b 0
)

REM No backup found - remove integration blocks using Python script
echo No backup found. Removing integration blocks...
where py >nul 2>&1
if not errorlevel 1 (
    call py -3 "%SCRIPT_DIR%remove_integration.py"
)

echo [OK] Qwen Image LoRA Loader integration removed.
pause
exit /b 0
```

**Portable Uninstaller (`uninstall_qwen_lora_portable.bat`)**:
```bat
echo Checking for backups...

REM Try new-style backup first
if exist "%NUNCHAKU_PATH%\__init__.py.qwen_image_backup" (
    copy "%NUNCHAKU_PATH%\__init__.py.qwen_image_backup" "%NUNCHAKU_PATH%\__init__.py" >nul
    echo Restored from backup: __init__.py.qwen_image_backup
    echo [OK] Qwen Image LoRA Loader integration removed.
    pause
    exit /b 0
)

REM No backup found - remove integration blocks using Python script
echo No backup found. Removing integration blocks...
"%PYTHON_CMD%" "%SCRIPT_DIR%remove_integration.py"

echo.
echo [OK] Qwen Image LoRA Loader integration removed.
pause
exit /b 0
```

**Key Changes**:
- **Line 26/50**: Checks if `.qwen_image_backup` file exists
- **Line 27/51**: Restores original `__init__.py` from backup
- **Line 29/53**: Exits immediately if backup restoration succeeds
- **Lines 35-39/59-60**: Fallback if no backup exists
- **Line 37**: Global uses `where py` + `call py -3`, Portable uses embedded Python directly
- **Line 38/60**: Calls `remove_integration.py` to remove blocks
- **Result**: Prefer backup restoration, fallback to marker-based removal
- **Key Change**: Removed `.backup` fallback - unified to `.qwen_image_backup` only

## Key Features

### 1. **Idempotent Installation**
- Running installer multiple times is safe
- Second run detects existing installation and skips
- No duplicate code, no manual intervention needed

### 2. **Backup Protection**
- Original `__init__.py` is backed up once
- Subsequent installs don't overwrite the backup
- Original state can always be restored

### 3. **Comprehensive Uninstallation**
- Two-stage approach: backup restoration or marker removal
- Handles cases with or without backup file
- Removes all duplicate blocks in a single operation

### 4. **User-Friendly Messaging**
- Clear messages about installation status
- Informative error handling
- No silent failures

## Testing

Tested scenarios:
1. ✅ Clean installation (first run)
2. ✅ Reinstallation (second run - skips)
3. ✅ Multiple reinstallations (skips all subsequent runs)
4. ✅ Uninstallation with backup
5. ✅ Uninstallation without backup
6. ✅ Reinstallation after uninstallation
7. ✅ Recovery from duplicate blocks

## Migration

**For users with v1.56 or earlier**:
- No action required if installation is working
- If you have duplicate blocks, run uninstaller then reinstaller
- All future installs will be protected against duplicates

**For new users**:
- Install normally using `install_qwen_lora.bat` or `install_qwen_lora_portable.bat`
- You can safely run the installer multiple times without issues

## Technical Benefits

1. **Reliability**: Eliminates user errors from duplicate installations
2. **Maintainability**: Cleaner code with proper state management
3. **Safety**: Original files are always recoverable
4. **Robustness**: Handles edge cases and manual modifications
5. **Professional**: Production-ready behavior expected in deployment scripts

## Related Issues

- Resolves duplicate installation issues
- Prevents backup file corruption
- Fixes uninstallation reliability

## Credits

- Implemented by: @ussoewwin
- Tested by: @ussoewwin
- Special thanks: Community feedback on installation issues

---

**Full Changelog**: See [README.md](README.md) for complete version history.
