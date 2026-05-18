"""Build RELEASE_NOTES/v2.4.4.md and zhmd/v2.4.4.md with language switchers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "RELEASE_NOTES"
ZHMD_DIR = ROOT / "zhmd"
EN_BODY_SRC = ROOT / "_release_v2.4.4_body_en.md"

EN_SWITCHER = """<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.4.4.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>"""

ZH_SWITCHER = """<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.4"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>"""

ZH_BODY = r"""## 概述

记录 v2.2.7（2026-01-07）新增的「首个 LoRA 重复文件读取消除」功能，在 v2.3.0（2026-01-22）AWQ 调制层 LoRA 修复提交中被无意删除，以及后续恢复的事实。

---

## 1. v2.2.7 中新增的功能

**提交**: `1ff4db9`
**版本**: v2.2.7 — LoRA 加载速度改进：消除重复文件读取

### 新增代码

```python
# OPTIMIZATION: Cache first LoRA state dict for reuse in processing loop
_cached_first_lora_state_dict = None
if lora_configs:
    first_lora_path_or_dict, first_lora_strength = lora_configs[0]
    first_lora_state_dict = _load_lora_state_dict(first_lora_path_or_dict)
    _cached_first_lora_state_dict = first_lora_state_dict  # Cache for reuse

# 1. Aggregate weights from all LoRAs
for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
    lora_name = lora_path_or_dict if isinstance(lora_path_or_dict, str) else "dict"
    # OPTIMIZATION: Reuse cached first LoRA state dict to avoid duplicate file I/O
    if idx == 0 and _cached_first_lora_state_dict is not None:
        lora_state_dict = _cached_first_lora_state_dict
    else:
        lora_state_dict = _load_lora_state_dict(lora_path_or_dict)
```

### 影响与意义

#### 为何首个 LoRA 会被读取两次

`compose_loras_v2()` 在两个独立阶段加载文件：

1. **调试日志阶段**（第 1199–1212 行）
   ```python
   first_lora_state_dict = _load_lora_state_dict(first_lora_path_or_dict)
   for key in first_lora_state_dict.keys():
       parsed_res = _classify_and_map_key(key)
       logger.info(...)
   # ← first_lora_state_dict 作为局部变量被丢弃
   ```

2. **实际处理循环阶段**（第 1217–1223 行）
   ```python
   for lora_path_or_dict, strength in lora_configs:
       lora_state_dict = _load_lora_state_dict(lora_path_or_dict)  # ← 同一文件再次读取
   ```

为调试日志加载的 `state_dict`（数百 MB 至数 GB）超出作用域后可被垃圾回收。实际处理循环会**从磁盘再次读取同一文件**，造成完全浪费。

#### 量化浪费

| 操作 | 修复前 | 修复后 | 减少 |
|------|--------|--------|------|
| **文件 I/O** | 2 次 | 1 次 | **50%** |
| **二进制解析与张量反序列化** | 2 次 | 1 次 | **50%** |
| **键分类（`_classify_and_map_key`）** | 2×N 次 | 1×N 次 | **50%** |

N = LoRA 文件中的键数量（通常为数百至数千）

#### 具体性能影响

- **500MB LoRA 文件**：约节省 1–3 秒（取决于磁盘速度）
- **2GB LoRA 文件**：约节省 5–10 秒
- **用户感知**：从执行提示到开始生成之间的等待时间缩短

#### 为何重要

1. **单 LoRA 是最常见用例**
   - 许多工作流只应用一个 LoRA
   - 此时「首个 LoRA」即唯一 LoRA，浪费为 100%

2. **文件 I/O 为同步阻塞**
   - `comfy.utils.load_torch_file()` 同步执行，完成前下游处理暂停
   - UI 线程被阻塞，ComfyUI 控制台看似「卡住」

3. **反序列化 CPU 密集**
   - safetensors / torch.load 的二进制解析带来高 CPU 负载
   - 大体积 LoRA（FLUX、SDXL、Qwen 类）尤为明显

4. **键分类为连续正则匹配**
   - `_classify_and_map_key()` 对每个键运行约 30 个正则模式
   - 成本随键数量线性增长

#### 缓存安全性

- `_cached_first_lora_state_dict` 为 `compose_loras_v2` 内局部变量
- 函数退出时自动丢弃，跨调用无状态泄漏
- `state_dict` 视为只读，调试日志不修改
- `idx == 0` 条件确保第二个及后续 LoRA 不受影响

---

## 2. v2.2.8 中保留

**提交**: `c113968`
**版本**: v2.2.8 — 修复 Issue #44：加载不支持的 LoRA 格式时控制台变慢

Issue #44（不支持 LoRA 格式产生大量日志）已修复，但 `_cached_first_lora_state_dict` 缓存机制原样保留。

```
git show c113968:nunchaku_code/lora_qwen.py | Select-String "cached_first_lora"
→ _cached_first_lora_state_dict = None
→ _cached_first_lora_state_dict = first_lora_state_dict
→ if idx == 0 and _cached_first_lora_state_dict is not None:
```

---

## 3. v2.3.0 中丢失

**提交**: `6498929`
**版本**: v2.3.0 — 通过运行时猴子补丁实现 AWQ 调制层 LoRA 修复
**变更范围**: `nunchaku_code/lora_qwen.py | 865 ++++++++++++++--------------- | 266 insertions(+), 599 deletions(-)`

### 丢失的代码（经 git diff 确认）

```diff
-        # OPTIMIZATION: Cache first LoRA state dict for reuse in processing loop
-        _cached_first_lora_state_dict = None
-        if lora_configs:
-            first_lora_path_or_dict, first_lora_strength = lora_configs[0]
-            first_lora_state_dict = _load_lora_state_dict(first_lora_path_or_dict)
-            _cached_first_lora_state_dict = first_lora_state_dict  # Cache for reuse

-        # 1. Aggregate weights from all LoRAs
-        for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
-            lora_name = lora_path_or_dict if isinstance(lora_path_or_dict, str) else "dict"
-            # OPTIMIZATION: Reuse cached first LoRA state dict to avoid duplicate file I/O
-            if idx == 0 and _cached_first_lora_state_dict is not None:
-                lora_state_dict = _cached_first_lora_state_dict
-            else:
-                lora_state_dict = _load_lora_state_dict(lora_path_or_dict)
+    # --- 4. Main Loading Loop ---
+    for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
+        if abs(strength) < 1e-5:
+            continue
```

### 丢失原因

`compose_loras_v2()` 分段重构时的迁移遗漏。

**v2.2.8 结构:**
```
缓存初始化 → 调试日志 → 循环（idx == 0 时复用缓存）
```

**v2.3.0 结构:**
```
# --- 1. Format Detection ---
# --- 2. Early Exit ---
# --- 3. Debug Inspection (First LoRA) ---
# --- 4. Main Loading Loop ---
```

第 3 节加载了 `first_lora_state_dict`，但为局部变量，第 4 节无法访问。第 4 节无条件调用 `_load_lora_state_dict_robust(lora_path_or_dict)`，首个 LoRA 被读取两次。

`enumerate(lora_configs)` 仍在，但缓存变量的声明、保存、复用三要素全部消失。

---

## 4. v2.4.3 时的代码

```python
# --- 3. Debug Inspection (First LoRA) ---
if lora_configs:
    first_lora_path_or_dict, first_lora_strength = lora_configs[0]
    first_lora_state_dict = _load_lora_state_dict_robust(first_lora_path_or_dict)
    _first_detection = _detect_lora_format(first_lora_state_dict)

# --- 4. Main Loading Loop ---
for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
    ...
    lora_state_dict = _load_lora_state_dict_robust(lora_path_or_dict)  # idx==0 时第二次读取
```

重复文件 I/O 再次出现。

---

## 5. 与 AWQ 猴子补丁的兼容性

AWQ 调制层 LoRA 修复（`_awq_lora_forward`）在 `_apply_lora_to_module` 内运行，位于 `compose_loras_v2` 流程**末端**：文件加载 → 权重处理 → 模块应用。

文件加载优化（缓存）位于流程**入口**。两者为独立层，可共存：

```python
_cached_first_lora_state_dict = None  # 新增
if lora_configs:
    first_lora_path_or_dict, first_lora_strength = lora_configs[0]
    first_lora_state_dict = _load_lora_state_dict_robust(first_lora_path_or_dict)
    _cached_first_lora_state_dict = first_lora_state_dict  # 新增
    _first_detection = _detect_lora_format(first_lora_state_dict)

for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
    ...
    if idx == 0 and _cached_first_lora_state_dict is not None:  # 新增
        lora_state_dict = _cached_first_lora_state_dict
    else:
        lora_state_dict = _load_lora_state_dict_robust(lora_path_or_dict)
```

对 AWQ 猴子补丁代码（第 2 节的 `should_apply_awq_mod` 检查、`_apply_lora_to_module` 内 AWQ 分支）**零影响**。缓存与 AWQ 补丁为独立层。

---

## 6. 迁移遗漏的事实

### 提交 6498929 的实际变更

```
nunchaku_code/lora_qwen.py | 865 ++++++++++++++--------------- | 266 insertions(+), 599 deletions(-)
```

- **删除 599 行**：`compose_loras_v2` 函数结构大规模重写
- **新增 266 行**：主要为 AWQ 猴子补丁（`_awq_lora_forward`、`DEFAULT_APPLY_AWQ_MOD_ENV` 等）

### 结构变化

**v2.2.8 及更早（顺序处理）:**

```python
def compose_loras_v2(...):
    _cached_first_lora_state_dict = None
    if lora_configs:
        first_lora = _load_lora_state_dict(...)
        _cached_first_lora_state_dict = first_lora
    for idx, (lora_path, strength) in enumerate(lora_configs):
        if idx == 0 and _cached_first_lora_state_dict is not None:
            lora_state_dict = _cached_first_lora_state_dict
        else:
            lora_state_dict = _load_lora_state_dict(lora_path)
```

**v2.3.0 及以后（分段）:**

```python
def compose_loras_v2(...):
    # --- 1. Z-Image / NextDiT Handling ---
    # --- 2. Environment Variable / Argument Logic for AWQ Mod ---
    # --- 3. Debug Inspection (First LoRA) ---
    if lora_configs:
        first_lora = _load_lora_state_dict_robust(...)
    # --- 4. Main Loading Loop ---
    for idx, (lora_path, strength) in enumerate(lora_configs):
        lora_state_dict = _load_lora_state_dict_robust(lora_path)
```

### 消失的三要素

| 要素 | v2.2.8 | v2.3.0 |
|------|--------|--------|
| **缓存变量声明** | `_cached_first_lora_state_dict = None` | 已删除 |
| **写入缓存** | `_cached_first_lora_state_dict = first_lora_state_dict` | 已删除 |
| **缓存复用** | `if idx == 0 and _cached_first_lora_state_dict is not None:` | 已删除 |

### 保留部分

```python
for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
```

`enumerate` 仍在，但因 `idx == 0` 分支逻辑消失，仅作计数、无法用于缓存复用。

### 提交说明中的事实

- 提交说明：`Implement AWQ modulation layer LoRA fix with Runtime Monkey Patch`
- 正文未提及「删除缓存」「移除优化」或「文件 I/O 变更」
- 目的是添加 AWQ 猴子补丁，并非有意丢弃文件 I/O 优化

### 结论

提交 6498929 将 `compose_loras_v2` 重构为分段结构时，保留了 `enumerate(lora_configs)`，但未将 `_cached_first_lora_state_dict` 三要素（声明、保存、复用）迁入新结构。提交目的是 AWQ 猴子补丁；删除文件 I/O 优化并非本意。

---

## 7. 恢复事实

### 修改文件

`nunchaku_code/lora_qwen.py`

### 新增代码

**第 3 节（Debug Inspection）:**

```python
    # --- 3. Debug Inspection (First LoRA) ---
    # OPTIMIZATION: Cache first LoRA state dict for reuse in processing loop
    _cached_first_lora_state_dict = None
    if lora_configs:
        first_lora_path_or_dict, first_lora_strength = lora_configs[0]
        first_lora_state_dict = _load_lora_state_dict_robust(first_lora_path_or_dict)
        _cached_first_lora_state_dict = first_lora_state_dict  # Cache for reuse
```

**第 4 节（Main Loading Loop）:**

```python
        # OPTIMIZATION: Reuse cached first LoRA state dict to avoid duplicate file I/O
        if idx == 0 and _cached_first_lora_state_dict is not None:
            lora_state_dict = _cached_first_lora_state_dict
        else:
            lora_state_dict = _load_lora_state_dict_robust(lora_path_or_dict)
```

### 意义

在 v2.3.0 AWQ 调制层猴子补丁完全兼容的前提下，恢复 v2.2.7「首个 LoRA 重复读取消除」功能。

- **第 3 节** 将 `first_lora_state_dict` 写入 `_cached_first_lora_state_dict`
- **第 4 节** 在 `idx == 0` 复用缓存，避免第二次 `_load_lora_state_dict_robust` 调用
- 文件 I/O、二进制解析、键分类减少 50%

对 AWQ 猴子补丁（`_awq_lora_forward`、`_apply_lora_to_module` 内 AWQ 分支）**零影响**。缓存位于文件加载层，AWQ 补丁位于模块应用层，完全独立。
"""


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    ZHMD_DIR.mkdir(parents=True, exist_ok=True)
    en_body = EN_BODY_SRC.read_text(encoding="utf-8").strip()
    (RELEASE_DIR / "v2.4.4.md").write_text(
        EN_SWITCHER + "\n\n" + en_body + "\n", encoding="utf-8"
    )
    (ZHMD_DIR / "v2.4.4.md").write_text(
        ZH_SWITCHER + "\n\n" + ZH_BODY.strip() + "\n", encoding="utf-8"
    )
    print("Wrote RELEASE_NOTES/v2.4.4.md and zhmd/v2.4.4.md")


if __name__ == "__main__":
    main()
