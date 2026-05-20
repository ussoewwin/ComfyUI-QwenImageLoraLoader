# Issue #25：ComfyUI 0.4.0 模型管理相关错误

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

- **问题现象**：ComfyUI 升级到 0.4.0 后，多个节点（部分环境下也包括本节点）出现 `TypeError: 'NoneType' object is not callable`、`AttributeError: 'NoneType' object has no attribute` 等错误。在我们使用的环境中，通过修改 ComfyUI 核心的 `model_management.py` 已解决上述错误。需要说明的是，在我们使用的环境中，**本节点（ComfyUI-QwenImageLoraLoader）并未出现这些错误**。请将 Nunchaku 库与 ComfyUI-Nunchaku 节点更新到最新版本。若已使用本节点（ComfyUI-QwenImageLoraLoader）的最新版本后错误仍然存在，则可能需要修改 ComfyUI 核心的 `model_management.py`。
- **根本原因**：ComfyUI 0.4.0 中，ComfyUI 核心的 `model_management.py` 缺少充分的 `None` 检查；在模型被卸载或垃圾回收后，对象变为 `None` 时仍访问其方法或属性，从而触发 `TypeError` 与 `AttributeError`。这不是单个节点的缺陷，而是 ComfyUI 0.4.0 模型管理（`model_management.py`）层面的结构性问题。
- **ComfyUI 0.4.0 的 GC 行为变化**：与 ComfyUI 0.3.x 相比，模型会更早被自动卸载，因此更容易出现如下流程：

```
ModelPatcher → GC → weakref(None)
```

这也解释了为何不同用户环境下问题是否出现会有差异。
- **技术依据**：
  1. **多处缺少 None 检查** — 这不是单个节点的 bug，而是核心主逻辑在访问 `None` 的属性时崩溃。像 `if model is None: continue` 这类修复属于 ComfyUI 核心本应在所有路径上具备的防御性代码。
  2. **未充分考虑弱引用后的 GC 行为** — ComfyUI 0.4.0 引入 `LoadedModel._model = weakref.ref(ModelPatcher)` 属于破坏性变更。弱引用目标被垃圾回收后会返回 `None`，但未做处理，面向破坏性更新的后续处理不完整。
  3. **多个节点呈连锁反应** — 这不是节点本身的问题，而是核心行为变更引发的连锁反应。模型加载/卸载、显存计算、GPU/CPU 卸载，以及 `ModelPatcher` 生命周期均由 ComfyUI 核心控制。
  4. **所有修复点均属于核心职责范围** — 修复位置（`model_memory`、`model_offloaded_memory`、`load_models_gpu`、`free_memory`、`model_unload`、`is_dead` 检查等）均为 ComfyUI 核心函数，不属于节点开发者应修改的范围。所有修复点都在核心逻辑内，除核心缺陷外难以解释。
  5. **应用修复后的结果** — 在我们使用的环境中，对 ComfyUI 核心的 `model_management.py` 增加 None 检查后，同类错误消失。说明通过补充核心本应具备的防御性代码可以解决问题。
- **模型生命周期与 ModelPatcher 初始化的关系**：
  - **事实 1：LoadedModel 与 ModelPatcher 的关系** — `LoadedModel` 类（ComfyUI 的 `model_management.py` 第 502–524 行）通过弱引用持有 `ModelPatcher`：

```python
def _set_model(self, model):
    self._model = weakref.ref(model)  # Weak reference to ModelPatcher

@property
def model(self):
    return self._model()  # Returns None when garbage collected
```

  - **事实 2：ModelPatcher 的初始化** — 在 `ModelPatcher` 类的 `__init__`（`model_patcher.py` 第 215–237 行）中，会初始化 `pinned` 属性：

```python
def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
    # ...
    self.pinned = set()  # Line 237: Initialized
```

  - **事实 3：ComfyUI 核心 `model_management.py` 中的修复内容** — 在 ComfyUI 核心的 `model_management.py` 中，会跳过 `model` 为 `None` 的 `LoadedModel` 实例：在 `load_models_gpu()` 中跳过 `model` 为 `None` 的 `LoadedModel`（第 712、727、743 行）；在 `free_memory()` 中排除 `model` 为 `None` 的 `LoadedModel`（第 646 行）。
  - **事实 4：修复前的问题** — `LoadedModel` 对 `ModelPatcher` 持有弱引用；被 GC 后，`LoadedModel.model` 返回 `None`。修复前仍对 `model` 为 `None` 的 `LoadedModel` 调用方法，导致错误。
  - **事实 5：修复后的行为** — 跳过 `model` 为 `None` 的 `LoadedModel` 后，错误不再出现，流程可正常继续。
  - **事实 6：为何 `copy.deepcopy` 会失败** — 被 deepcopy 的字典中仍保留指向已被 GC 的 `ModelPatcher` 的引用；访问这些引用时得到 `None`，导致 deepcopy 中止。
  - **事实 7：确认项** — 在我们使用的环境中应用修复后，`copy.deepcopy` 与 `pinned` 属性相关错误不再出现。请将 Nunchaku 库与 ComfyUI-Nunchaku 节点更新到最新版本，但这可能仍不足。在我们使用的环境中，本节点（ComfyUI-QwenImageLoraLoader）虽未出现这些错误，但对 ComfyUI 核心 `model_management.py` 的修复可能间接降低了其他节点出错的概率。
- **重要说明：并非 Nunchaku 库的问题** — 本问题并非 Nunchaku 库实现导致。Nunchaku 的 `model_config` 与 `ModelPatcher` 本身正常。问题出在上游，即 ComfyUI 核心的 `model_management.py` 的 GC 处理。
- **推测（可能因环境而异）**：修复后 `ModelPatcher` 可正常完成初始化，从而 `pinned` 也会被正确初始化；在 `__del__` 中访问 `self.pinned` 不会报错。
- **建议**：
  1. 将 Nunchaku 库与 ComfyUI-Nunchaku 节点更新到最新版本（可缓解 `model_config` 相关问题）
  2. 考虑对 ComfyUI 核心的 `model_management.py` 应用 None 检查类修复（可能从根因上缓解）
- **说明**：这是第一层支持措施。我们已在环境中对 ComfyUI 核心的 `model_management.py` 所应用的修复技术细节已单独整理发布。详见 [COMFYUI_0.4.0_UPDATE_ERROR_FIXES.md](./COMFYUI_0.4.0_UPDATE_ERROR_FIXES.md)。请注意，这些修复是在特定环境中实施的，未必适用于所有环境；也可能同时缓解 `copy.deepcopy` 与 `pinned` 属性相关错误。
