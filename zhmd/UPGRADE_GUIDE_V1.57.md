# v1.57 及更早版本用户升级指南

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/UPGRADE_GUIDE_V1.57.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

若您在 ComfyUI-nunchaku 的 `__init__.py` 中安装过 **v1.57 或更早版本** 的集成代码，可按下列四种方式之一处理。

## 选项 1：保持现状（推荐大多数用户）

**旧集成代码可继续工作，无需额外操作：**

1. 将 ComfyUI-QwenImageLoraLoader 更新到 **v1.60**：
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader
   git pull origin main
   ```

2. 重启 ComfyUI

**完成。** 旧集成代码会被安全忽略，新的独立加载机制会接管。现有工作流与 LoRA 文件可照常使用，行为与升级前一致。

## 选项 2：清理旧集成代码（保持仓库整洁）

**⚠️ 仅 Windows — macOS / Linux 无批处理脚本**

若要从 ComfyUI-nunchaku 的 `__init__.py` 中移除旧集成代码：

**本选项仅适用于 Windows。** 若使用 macOS 或 Linux，请使用下方 **选项 3（手动清理）**。

### Windows 用户

1. 先更新到 **v1.60**：
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader
   git pull origin main
   ```

2. 运行卸载脚本，移除旧集成代码：

   **全局 Python 环境：**
   ```cmd
   cd ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader
   uninstall_qwen_lora.bat
   ```

   **便携版 ComfyUI（内嵌 Python）：**
   ```cmd
   cd ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader
   uninstall_qwen_lora_portable.bat
   ```

3. 卸载程序会将 ComfyUI-nunchaku 的 `__init__.py` 恢复为原始状态

4. 重启 ComfyUI

**卸载 旧集成代码后，节点仍可正常工作** — v1.60 起使用独立加载机制，不依赖 nunchaku 内的集成块。

## 选项 3：手动清理（macOS / Linux，或偏好手动编辑的用户）

**适用于 macOS / Linux 用户**（无批处理脚本），或希望手动编辑文件的用户。

### 方法 A：手动删除集成代码块

1. 用文本编辑器打开：`ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py`

2. 找到并删除文件末尾的整段代码块（如下所示）：

```python
# BEGIN ComfyUI-QwenImageLoraLoader Integration
try:
    # Import from the independent ComfyUI-QwenImageLoraLoader
    import sys
    import os
    qwen_lora_path = os.path.join(os.path.dirname(__file__), "..", "ComfyUI-QwenImageLoraLoader")
    if qwen_lora_path not in sys.path:
        sys.path.insert(0, qwen_lora_path)
    
    # Import directly from the file path
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwenimage", os.path.join(qwen_lora_path, "nodes", "lora", "qwenimage.py"))
    qwenimage_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwenimage_module)
    
    NunchakuQwenImageLoraLoader = qwenimage_module.NunchakuQwenImageLoraLoader
    NunchakuQwenImageLoraStack = qwenimage_module.NunchakuQwenImageLoraStack

    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = qwenimage_module.NunchakuQwenImageLoraStack
    logger.info("Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader")
except ImportError:
    logger.exception("Nodes `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` import failed:")
# END ComfyUI-QwenImageLoraLoader Integration
```

3. 删除从 `# BEGIN ComfyUI-QwenImageLoraLoader Integration` 到 `# END ComfyUI-QwenImageLoraLoader Integration` 的**整段**（含首尾标记行）

4. 保存文件

5. 重启 ComfyUI

**重要：** 务必按 BEGIN / END 标记定位；删除范围须包含上述两行标记及其之间的全部内容。

## 选项 4：恢复官方 ComfyUI-nunchaku `__init__.py`（紧急恢复）

**若 ComfyUI-nunchaku 的 `__init__.py` 已损坏、无法使用或无法恢复**，可从官方 Nunchaku 仓库恢复：

1. 从 [ComfyUI-nunchaku 仓库](https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/__init__.py) 下载官方 `__init__.py`

2. 覆盖到：`ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py`

3. 重启 ComfyUI

官方 `__init__.py` 不含 ComfyUI-QwenImageLoraLoader 集成代码。v1.60 仍可正常工作，因其使用独立加载机制。

## 为何不再需要集成代码

自 **v1.60** 起，ComfyUI-QwenImageLoraLoader 作为**完全独立的自定义节点**运行。原因如下：

1. **ComfyUI 自动发现节点**：启动时自动扫描 `custom_nodes/` 并加载各插件的 `__init__.py`

2. **自动合并 NODE_CLASS_MAPPINGS**：各插件的节点映射在启动时自动合并到统一注册表

3. **直接类型导入**：加载器直接从 nunchaku 包导入 `NunchakuQwenImageTransformer2DModel`，无需主程序集成

4. **与模型无关的 LoRA 合成**：`compose_loras_v2()` 适用于任意带 `_lora_slots` 的模型，不依赖主程序

5. **基于 Wrapper 的架构**：LoRA 逻辑由自包含的 `ComfyQwenImageWrapper` 处理

完整技术说明（共 7 章）见 GitHub：[v1.60 发行说明](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.60)。

## 向后兼容

✅ **现有环境可继续使用：**

- v1.57 及更早版本创建的工作流无需修改
- LoRA 文件无需任何改动
- ComfyUI-nunchaku `__init__.py` 中的旧集成代码会被安全忽略
- 节点输入/输出无破坏性变更
