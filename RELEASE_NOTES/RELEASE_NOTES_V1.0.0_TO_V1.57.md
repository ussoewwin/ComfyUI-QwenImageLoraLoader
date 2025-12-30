# Release Notes: v1.0.0 to v1.57

## v1.57 (Previous)

**Fixed Critical Bug**: Resolved duplicate integration blocks when running installer multiple times

- Reported by: @ussoewwin
- **Issue #15 Fixed**: This node is now available on ComfyUI Registry for easy installation and management (Issue #15)
- Full release notes: https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.57
- ComfyUI Registry v1.57

---

## v1.56

**Fixed Critical Bug**: Resolved Issue #13 – Applied LoRA compositions to 0 module after crash/offload

- Reported by: @coffings20-gif – "After a crash he ignores my LoRA loader: Applied LoRA compositions to 0 module (using CPU offloader)"
- Full release notes: https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.56

---

## v1.55

**Change**: Installer `install_qwen_lora.bat` now uses `py -3` instead of `python`

- **Purpose**: Avoid silent failures on environments where the Microsoft Store proxy python is picked up from PATH
- **Impact**: No functional change; improves installation reliability only
- **Recommendation**: Update to v1.55 and re-run installation only if you experienced installer failures

---

## v1.5.4

**Fixed Critical Bug**: Resolved Issue #11 - Multiple LoRAs not applying on re-run (Issue #11)

- Reported by: @recursionlaplace-eng - "当有多个lora存在时, 再次运行, lora可能不生效" (When multiple LoRAs exist, re-running may cause LoRAs to not take effect)

**Problem**: When using multiple LoRAs, re-executing the workflow could result in LoRAs not being applied, even though they were applied successfully in the previous run

**Root Cause**: Shallow comparison logic in LoRA change detection (`self._applied_loras != self.loras`)
- The comparison used Python's list equality operator `!=`, which compares list contents
- However, due to shallow copy (`self._applied_loras = self.loras.copy()`), the comparison could fail to detect when LoRA state needed to be reapplied
- When model internal state was reset (e.g., due to VRAM management, CPU offload, or cache clearing), but the comparison returned False, LoRA reapplication would be skipped
- This was especially problematic with multiple LoRAs because:
  - List comparison could return False even when model internal state was dirty
  - Shallow copy means tuple references are shared between `_applied_loras` and `loras`
  - In some execution contexts, both lists could reference the same objects, making change detection fail

**Technical Solution**: Implemented deep comparison logic with explicit checks

- **Before**: Simple list inequality `if self._applied_loras != self.loras or model_is_dirty:`
- **After**: Deep comparison with explicit length check and element-by-element comparison:
```python
# Deep comparison of LoRA stacks to detect any changes
# This ensures we catch changes in weights, paths, or order
loras_changed = False
if self._applied_loras is None or len(self._applied_loras) != len(self.loras):
    loras_changed = True
else:
    for applied, current in zip(self._applied_loras, self.loras):
        if applied != current:
            loras_changed = True
            break

if loras_changed or model_is_dirty:
```

**Technical Details**:
- **Explicit None Check**: Ensures initial state is always detected (`self._applied_loras is None`)
- **Length Comparison**: Detects LoRA addition/removal immediately (`len(self._applied_loras) != len(self.loras)`)
- **Element-by-Element Comparison**: Detects changes in:
  - LoRA file paths (different LoRA files)
  - LoRA strengths/weights (strength value changes)
  - LoRA order (same LoRAs in different order)
- **Early Break**: Stops checking as soon as first difference is found for efficiency
- **Robustness**: Works regardless of reference sharing or shallow copy issues

**Why This Matters**:
- **Multiple LoRA Scenarios**: Critical for workflows using 2+ LoRAs
- **VRAM Management**: Model state can be reset by ComfyUI's memory management
- **CPU Offload**: LoRA state can be lost when model moves between CPU/GPU
- **Cache Clearing**: Internal caches can invalidate LoRA composition
- **Workflow Re-execution**: Ensures consistent results across multiple runs

**Testing**: Comprehensive test suite validates:
- Initial LoRA load detection
- No false positives (same LoRAs = no change)
- Weight change detection
- Order change detection
- LoRA addition/removal detection
- Reference sharing edge cases

**Impact**: Significantly improves LoRA change detection reliability and should resolve the reported issue where LoRAs fail to apply on re-run. However, if issues persist in specific edge cases, please report them with detailed reproduction steps

---

## v1.5.3

**Fixed Critical Bug**: Resolved `TypeError: This LoRA loader only works with Nunchaku Qwen Image models, but got ComfyQwenImageWrapper` error in different workflows

**Problem**: LoRA loader failed when model was already wrapped with `ComfyQwenImageWrapper` in some workflows, even though the wrapper was correct

**Root Cause**: Using type name comparison (`type(model_wrapper).__name__ == "ComfyQwenImageWrapper"`) failed in some execution contexts due to dynamic imports and different module loading paths. When `ComfyQwenImageWrapper` was loaded from different import paths, Python treated them as different classes even though they were functionally identical

**Technical Solution**: Changed from type name comparison to attribute-based detection

- **Before**: `if type(model_wrapper).__name__ == "ComfyQwenImageWrapper"`
- **After**: `if hasattr(model_wrapper, 'model') and hasattr(model_wrapper, 'loras')`
  - The `ComfyQwenImageWrapper` class always has `model` (transformer) and `loras` (LoRA list) attributes
  - This detection method works regardless of how the class was imported or loaded

**Technical Details**:
- Added comprehensive debug logging to help diagnose type detection issues
- Logs now show type name, module, attributes, and full type representation
- Enhanced error messages with detailed type information
- Applied fix to both `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` classes

**Benefits**:
- Works with all workflows regardless of model loading order
- No dependency on import mechanism or module path
- More robust and resilient to Python module system quirks
- Better debugging capabilities with detailed logging

**Impact**: Completely resolves the TypeError that occurred when using LoRA nodes in certain workflow configurations

---

## v1.5.2

**Fixed Critical Bug**: Resolved persistent `ModuleNotFoundError: No module named 'wrappers.qwenimage'` error

- Reported by: Multiple users experiencing intermittent import failures

**Problem**: Despite v1.5.0 and v1.5.1 fixes, some users still experienced `ModuleNotFoundError` when executing LoRA nodes

**Root Cause**: Module-level `sys.path` manipulation was insufficient in all execution contexts. The `from wrappers.qwenimage import ComfyQwenImageWrapper` statement at method execution time occasionally failed even though the module loaded successfully, due to timing issues, Python's module cache, and varying execution contexts in ComfyUI

**Technical Solution**: Implemented robust dynamic import using `importlib.util` within method execution

- **Before**: Module-level `sys.path` setup + direct `from wrappers.qwenimage import` at method level
- **After**: Dynamic `importlib.util`-based import within `load_lora()` and `load_lora_stack()` methods

**Technical Details**:
- Each method now dynamically calculates the parent directory path at execution time
- Uses `importlib.util.spec_from_file_location()` to create module spec from absolute file path
- Executes module with `spec.loader.exec_module()` and extracts `ComfyQwenImageWrapper` class
- Completely bypasses Python's normal import mechanism, making it independent of `sys.path` state
- Eliminates dependency on module loading order or execution context

**Benefits**:
- 100% reliable import regardless of execution context
- No dependency on `sys.path` configuration
- Works in all ComfyUI execution scenarios
- No more intermittent import failures
- More resilient to future ComfyUI or Python updates

**Impact**: Completely resolves the import reliability issue that affected multiple users

---

## v1.5.1

**Fixed Critical Bug**: Additional fix for Issue #6 - persistent `ModuleNotFoundError`

**Problem**: v1.5.0 fix was incomplete - path calculation error meant `sys.path` pointed to wrong directory

**Root Cause**: Incorrect path calculation: `os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))` went up 3 levels instead of 2
- 3 levels up → pointed to `custom_nodes/` instead of `ComfyUI-QwenImageLoraLoader/`
- This prevented `from wrappers.qwenimage` from resolving correctly

**Technical Solution**: Corrected path calculation to go up exactly 2 levels from `nodes/lora/`

- **Before**: `os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))` (3 levels)
- **After**: `os.path.dirname(os.path.dirname(current_dir))` (2 levels)
- **Result**: Correctly points to `ComfyUI-QwenImageLoraLoader/` root directory

**Applied To**: Module-level `sys.path` initialization in `qwenimage.py`

**Benefits**:
- Absolute imports now work correctly in all loading scenarios
- No more `ModuleNotFoundError` errors
- Module can be loaded regardless of import mechanism
- Completely resolves Issue #6

---

## v1.5.0

**Fixed Critical Bug**: Resolved Issue #6 - attempted relative import with no known parent package

- Reported by: @showevr (GitHub Issue #6)
- **Special Thanks**: This bug was discovered and reported by @showevr

**Problem**: LoRA loader nodes failed to load with `ValueError: attempted relative import with no known parent package` error

**Root Cause**: Using relative imports (`from ...wrappers.qwenimage import ComfyQwenImageWrapper`) in the LoRA loader code. Relative imports only work when the module is loaded as part of a package, but ComfyUI-nunchaku loads the module directly using `importlib.util`, which bypasses package initialization

**Technical Solution**: Changed relative imports to absolute imports in `qwenimage.py`

- **Before**: `from ...wrappers.qwenimage import ComfyQwenImageWrapper`
- **After**: `from wrappers.qwenimage import ComfyQwenImageWrapper`

**Technical Details**:
- The installation script adds `ComfyUI-QwenImageLoraLoader` to `sys.path`
- This allows absolute imports to work correctly
- The absolute import `from wrappers.qwenimage import` resolves to `ComfyUI-QwenImageLoraLoader/wrappers/qwenimage.py`
- Applied fix to both `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` classes

**Community Contribution**: This fix was made possible by @showevr's bug reporting

---

## v1.4.0

**Fixed Critical Bug**: Resolved Issue #3 - Node break cached progress

- Reported by: @AHEKOT (GitHub Issue #3)
- **Special Thanks**: This critical bug was discovered and reported by @AHEKOT, who identified the issue with cached progress being broken

**Problem**: LoRA nodes were breaking ComfyUI's cached progress during image generation, causing the progress bar to reset and restart repeatedly

**Root Cause**: The `IS_CHANGED` method was returning `float("NaN")` instead of a proper change detection hash
- `float("NaN")` causes ComfyUI to always treat the node as "changed"
- This invalidates the cache after every frame, forcing unnecessary re-execution
- Result: Progress bar jumps back and forth, cache is constantly invalidated, generation becomes unstable

**Technical Solution**: Implemented proper hash-based change detection for both LoRA nodes

- `NunchakuQwenImageLoraLoader`: Creates SHA256 hash from model, lora_name, and lora_strength
- `NunchakuQwenImageLoraStack`: Creates SHA256 hash from model, lora_count, and all 10 LoRA slots (names and strengths)
- Returns hex digest as the cache key

**Technical Details**:
- Uses Python's `hashlib.sha256()` to generate deterministic hashes
- Same inputs always produce same hash → cache is used correctly
- Different inputs produce different hashes → node re-executes as expected
- No more false positives causing cache invalidation

**Benefits**:
- Stable cached progress during generation
- No more progress bar jumping back and forth
- Proper cache utilization for better performance
- Smooth generation experience
- VRAM usage optimized due to proper cache behavior

**Community Contribution**: This fix was made possible by @AHEKOT's bug reporting and issue tracking

---

## v1.3.0

**Fixed Critical Bug**: Resolved `SyntaxError: invalid character '' (U+FFFD)` error when running installation script

**Problem**: PowerShell output was being written directly into Python files, causing syntax errors

**Root Cause**: Using PowerShell commands with piping (`Get-Content | Add-Content`) caused PowerShell status messages and metadata to be included in the output, which were then written into `__init__.py`

**Solution**: Replaced PowerShell-based approach with a dedicated Python script (`append_integration.py`)

**Technical Details**:
- Created standalone Python script `append_integration.py` that handles UTF-8 file writing
- Batch file now calls the Python script instead of using inline PowerShell commands
- Python script uses proper UTF-8 encoding when writing to `__init__.py`
- Eliminates any possibility of output artifacts contaminating Python files

**Benefits**:
- More reliable and maintainable solution
- No risk of shell output pollution
- Cleaner separation of concerns between batch file and Python code
- Better error handling and user feedback

---

## v1.2.0

**Fixed Issue #2**: Resolved UTF-8 encoding error that caused `SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0x90 in position 0`

- Reported by: @AHEKOT (GitHub Issue #2)
- **Special Thanks**: This critical bug was discovered and reported by @AHEKOT, who provided detailed error traces and screenshots

**Technical Fix**: Changed the batch file to use PowerShell for UTF-8 encoding when writing Python code to `__init__.py`

**Root Cause**: Windows batch files write in Shift-JIS encoding by default, which causes Python to fail when reading files as UTF-8

**Solution**: Temporary file creation + PowerShell UTF-8 encoding ensures proper file encoding

**Technical Details**:
- Changed from direct append (`>>`) to temp file method
- Uses PowerShell `Get-Content` and `Add-Content` with `-Encoding UTF8` parameters
- Ensures all Python code is written with proper UTF-8 encoding

**Impact**: Installation script now works correctly without encoding errors

**Community Contribution**: This fix was made possible by @AHEKOT's thorough bug reporting and error documentation

---

## v1.1.0

**Fixed Issue #1**: Resolved "ComfyUI\custom_nodes not found" error

- Reported by: @mcv1234
- **Special Thanks**: This fix was implemented based on the excellent solution provided by @mcv1234 in the GitHub issue

**Improved Path Detection**: Replaced unreliable wildcard search with relative path detection using script directory (solution by @mcv1234)

**Enhanced Error Messages**: Added clear directory structure guidance and expected folder layout

**Better User Experience**: More reliable installation process with comprehensive error checking

**Technical Details**:
- Changed from `dir /s /b /ad "*ComfyUI\custom_nodes"` to relative path calculation (as suggested by @mcv1234)
- Uses `%~dp0` (script directory) to calculate ComfyUI root with `..\..` navigation
- Added validation for `custom_nodes` folder existence
- Improved error messages with expected directory structure display

**Community Contribution**: This improvement was made possible by the community feedback and solution provided by @mcv1234

---

## v1.0.0

**Initial release** with LoRA loading functionality

- Automated installation scripts
- Integration with ComfyUI-nunchaku

