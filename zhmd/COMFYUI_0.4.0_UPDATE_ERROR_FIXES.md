# ComfyUI 0.4.0 更新：model_management.py 中多种错误的修复

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/COMFYUI_0.4.0_UPDATE_ERROR_FIXES.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

## 错误的根本原因

在 ComfyUI 的 `model_management.py` 中，模型被卸载或垃圾回收后，代码访问了 `None` 对象上的方法或属性，导致 `TypeError` 和 `AttributeError`。

## 修改的文件

**文件**：`ComfyUI/comfy/model_management.py`

## 详细修复内容

### 修复 1：`is_dead()` 方法（第 597–600 行）

**错误原因：**
```python
# 修改前
def is_dead(self):
    return self.real_model() is not None and self.model is None
```
`model_unload()` 将 `self.real_model` 设为 `None`，但 `is_dead()` 调用了 `self.real_model()`，导致 `TypeError: 'NoneType' object is not callable`。

**修改后：**
```python
def is_dead(self):
    if self.real_model is None:
        return False
    return self.real_model() is not None and self.model is None
```

**含义：**
如果 `self.real_model` 为 `None`，返回 `False`；否则将其作为 `weakref.ref` 调用。

---

### 修复 2：`model_memory()` 方法（第 526–529 行）

**错误原因：**
```python
# 修改前
def model_memory(self):
    return self.model.model_size()
```
当 `self.model` 为 `None`（已被垃圾回收）时调用 `model_size()`，导致 `AttributeError: 'NoneType' object has no attribute 'model_size'`。

**修改后：**
```python
def model_memory(self):
    if self.model is None:
        return 0
    return self.model.model_size()
```

**含义：**
如果 `self.model` 为 `None`，返回 `0`（已释放，因此内存使用量为 0）。

---

### 修复 3：`model_loaded_memory()` 方法（第 531–534 行）

**错误原因：**
```python
# 修改前
def model_loaded_memory(self):
    return self.model.loaded_size()
```
当 `self.model` 为 `None` 时调用 `loaded_size()` 会导致错误。

**修改后：**
```python
def model_loaded_memory(self):
    if self.model is None:
        return 0
    return self.model.loaded_size()
```

**含义：**
如果 `self.model` 为 `None`，返回 `0`。

---

### 修复 4：`model_offloaded_memory()` 方法（第 536–539 行）

**错误原因：**
```python
# 修改前
def model_offloaded_memory(self):
    return self.model.model_size() - self.model.loaded_size()
```
当 `self.model` 为 `None` 时调用 `model_size()` 会导致错误。

**修改后：**
```python
def model_offloaded_memory(self):
    if self.model is None:
        return 0
    return self.model.model_size() - self.model.loaded_size()
```

**含义：**
如果 `self.model` 为 `None`，返回 `0`。

---

### 修复 5：`model_memory_required()` 方法（第 541–547 行）

**错误原因：**
```python
# 修改前
def model_memory_required(self, device):
    if device == self.model.current_loaded_device():
        return self.model_offloaded_memory()
    else:
        return self.model_memory()
```
当 `self.model` 为 `None` 时调用 `current_loaded_device()` 会导致错误。

**修改后：**
```python
def model_memory_required(self, device):
    if self.model is None:
        return 0
    if device == self.model.current_loaded_device():
        return self.model_offloaded_memory()
    else:
        return self.model_memory()
```

**含义：**
如果 `self.model` 为 `None`，返回 `0`。

---

### 修复 6：`model_unload()` 方法（第 574–583 行）

**错误原因：**
```python
# 修改前
def model_unload(self, memory_to_free=None, unpatch_weights=True):
    # ...
    self.model.detach(unpatch_weights)
    self.model_finalizer.detach()  # ← 此处出错
    self.model_finalizer = None
    self.real_model = None
    return True
```
当 `self.model_finalizer` 为 `None` 时调用 `detach()`，导致 `AttributeError: 'NoneType' object has no attribute 'detach'`。

**修改后：**
```python
def model_unload(self, memory_to_free=None, unpatch_weights=True):
    # ...
    self.model.detach(unpatch_weights)
    if self.model_finalizer is not None:
        self.model_finalizer.detach()
    self.model_finalizer = None
    self.real_model = None
    return True
```

**含义：**
仅在 `self.model_finalizer` 不为 `None` 时调用 `detach()`。

---

### 修复 7：`load_models_gpu()` 方法 — 克隆检测（第 710–722 行）

**错误原因：**
```python
# 修改前
for loaded_model in models_to_load:
    to_unload = []
    for i in range(len(current_loaded_models)):
        if loaded_model.model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload
    for i in to_unload:
        model_to_unload = current_loaded_models.pop(i)
        model_to_unload.model.detach(unpatch_all=False)
        model_to_unload.model_finalizer.detach()  # ← 错误
```
当 `loaded_model.model` 或 `model_to_unload.model_finalizer` 为 `None` 时调用方法会导致错误。

**修改后：**
```python
for loaded_model in models_to_load:
    if loaded_model.model is None:
        continue
    to_unload = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].model is not None and loaded_model.model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload
    for i in to_unload:
        model_to_unload = current_loaded_models.pop(i)
        if model_to_unload.model is not None:
            model_to_unload.model.detach(unpatch_all=False)
        if model_to_unload.model_finalizer is not None:
            model_to_unload.model_finalizer.detach()
```

**含义：**
- 如果 `loaded_model.model` 为 `None`，跳过
- 如果 `current_loaded_models[i].model` 为 `None`，不调用 `is_clone()`
- 如果 `model_to_unload.model` 为 `None`，不调用 `detach()`
- 如果 `model_to_unload.model_finalizer` 为 `None`，不调用 `detach()`

---

### 修复 8：`load_models_gpu()` 方法 — 显存计算（第 724–727 行）

**错误原因：**
```python
# 修改前
total_memory_required = {}
for loaded_model in models_to_load:
    total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)
```
如果 `loaded_model.model` 为 `None`，`model_memory_required()` 可能出错。

**修改后：**
```python
total_memory_required = {}
for loaded_model in models_to_load:
    if loaded_model.model is not None:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)
```

**含义：**
从显存计算中排除 `loaded_model.model` 为 `None` 的模型（已被垃圾回收）。

---

### 修复 9：`load_models_gpu()` 方法 — 模型加载（第 740–744 行）

**错误原因：**
```python
# 修改前
for loaded_model in models_to_load:
    model = loaded_model.model
    torch_dev = model.load_device  # ← 错误
```
当 `loaded_model.model` 为 `None` 时访问 `load_device` 会导致错误。

**修改后：**
```python
for loaded_model in models_to_load:
    model = loaded_model.model
    if model is None:
        continue
    torch_dev = model.load_device
```

**含义：**
如果 `model` 为 `None`，跳过。

---

### 修复 10：`free_memory()` 函数（第 641–647 行）

**错误原因：**
```python
# 修改前
for i in range(len(current_loaded_models) -1, -1, -1):
    shift_model = current_loaded_models[i]
    if shift_model.device == device:
        if shift_model not in keep_loaded and not shift_model.is_dead():
            can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
```
当 `shift_model.model` 为 `None` 时调用 `sys.getrefcount(shift_model.model)` 会导致错误。将 `model` 为 `None` 的模型添加到 `can_unload` 还会导致显存计算不准确。

**修改后：**
```python
for i in range(len(current_loaded_models) -1, -1, -1):
    shift_model = current_loaded_models[i]
    if shift_model.device == device:
        if shift_model not in keep_loaded and not shift_model.is_dead():
            if shift_model.model is not None:
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False
```

**含义：**
从 `can_unload` 中排除 `shift_model.model` 为 `None` 的模型（已被垃圾回收，因此从显存计算中排除）。

---

## 修复效果

1. **错误预防**：`None` 检查可预防 `TypeError` 和 `AttributeError`
2. **更准确的显存计算**：排除 `model` 为 `None` 的模型可提高计算准确性
3. **更好的显存管理**：跳过不必要的模型可优化显存使用
4. **更快的处理速度**：减少回退到平铺编码的情况，启用正常处理

这些修复稳定了 ComfyUI 的模型管理，并减少了与显存相关的错误。

---

# 修复：由弱引用模型生命周期引起的 model_management.py 中多个 NoneType 错误的完整修复方案（10 个关键位置）

## 概要

本 PR 修复了 ComfyUI 0.4.0 中因弱引用模型生命周期管理而在多个节点出现的 NoneType 错误。

根本原因是 `LoadedModel.model` 在底层 `ModelPatcher` 被垃圾回收后可能会返回 `None`——但若干核心函数未对此情况进行保护。

这会导致不可预测的崩溃，例如：

- `TypeError: 'NoneType' object is not callable`
- `AttributeError: 'NoneType' object has no attribute 'weights'`
- `AttributeError: 'NoneType' object has no attribute '__deepcopy__'`

我在 `model_management.py` 中识别出 10 个关键位置，需要在引用 `lm.model` 之前检查 `None` 值。

应用这些检查后，我所在环境中的所有崩溃均已消除，同时也解决了多个用户报告的其他节点故障（例如 Issue #25）。

## 技术根本原因

### 1. 基于弱引用的模型所有权（0.4.0 变更）

ComfyUI 0.4.0 引入了：

```python
self._model = weakref.ref(model)
```

**含义：**

`LoadedModel.model → 在 GC 后返回 None`

然而，若干核心函数假设模型始终存在。

### 2. 核心内存/模型例程中缺失 None 检查

受影响的函数包括：

- `is_dead()`
- `model_memory()`
- `model_loaded_memory()`
- `model_offloaded_memory()`
- `model_memory_required()`
- `model_unload()`
- `load_models_gpu()`（3 个位置）
- `free_memory()`

所有这些函数都会遍历 `LoadedModel` 对象并引用 `lm.model`，但未确认其非 `None`。

当 GC 在卸载过程中回收 `ModelPatcher` 时，这会导致不相关节点中直接崩溃。

### 3. 为何这会导致广泛的节点故障

当 `LoadedModel.model` 变为 `None` 时，以下操作会在整个系统中失败：

- 显存估算
- 模型固定
- 深拷贝
- 设备切换
- 卸载/加载决策

这会产生数十种不同的堆栈跟踪，具体取决于执行时机，使得单个节点看起来有缺陷。

实际上，问题源自引用已回收 `ModelPatcher` 的陈旧 `LoadedModel` 对象。

## 修复方案

本 PR 在所有受影响的位置添加了明确的：

```python
if lm.model is None:
    continue
```

或等效的提前返回。

应用补丁后：

- ✔ 无残留 NoneType 崩溃
- ✔ 内存管理行为确定性
- ✔ 模型加载/卸载循环不再污染缓存状态
- ✔ 第三方节点不再因副作用而失败
- ✔ 用户报告的 issue #25（及类似问题）不再复现

## 为何此补丁是必要的

基于弱引用的模型所有权在设计上是正确的。

然而：

**任何弱引用解引用都必须将 `None` 视为有效且预期的状态。**

该不变式当前在若干核心路径中被违反。

本 PR 确保：

- 垃圾回收后的模型不再破坏模型管理管线
- 所有内存和生命周期操作对 `None` 具有鲁棒性
- 即使在底层对象被销毁时，`LoadedModel` 仍然安全

我可以提供：

- 完整的 diff
- 额外测试（如需）
- 经验证证明补丁后系统稳定性的日志

感谢您为 ComfyUI 所做的工作——此补丁应能大幅提升 0.4.0 模型管理系统的稳定性。

---

**注**：上述 PR 提案源自本文件中记录的 10 项修复。这些修复已在我所在环境中应用以解决 NoneType 错误，此提案基于这些发现总结技术分析和建议。

---

有关症状及 Nunchaku / 节点更新的背景，请参见 [Issue #25：ComfyUI 0.4.0 模型管理错误](COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md)。
