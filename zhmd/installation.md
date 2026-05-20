# 安装与功能

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/installation.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

## 🎉 重大更新：v1.60 — 简化安装（无需集成！）

**自 v1.60 起，不再需要在 ComfyUI-nunchaku 的 `__init__.py` 中手动集成。**

本节点现为完全独立的自定义节点，开箱即用。只需将本仓库克隆到 `custom_nodes` 目录并重启 ComfyUI，节点会通过 ComfyUI 内置的自动节点加载机制出现在节点菜单中。

### v1.60 的变更
- ✅ **移除对 ComfyUI-nunchaku 集成的依赖** — LoRA 加载器现为独立插件
- ✅ **简化安装** — 无需批处理脚本或手动编辑文件
- ✅ **更清晰的架构** — 节点注册自动完成
- ✅ **向后兼容** — 现有 LoRA 文件与工作流可继续使用

如需了解为何不再需要集成，请参阅 [v1.60 发行说明](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.60)。

## 功能

- **NunchakuQwenImageLoraLoader**：为 Qwen Image 模型加载并应用单个 LoRA
- **NunchakuQwenImageLoraStack**：通过动态 UI 应用多个 LoRA
- **NunchakuQwenImageDiffsynthControlnet**：将 diffsynth ControlNet 应用于 Nunchaku Qwen Image 模型（v2.0）
- **动态显存管理**：根据可用显存自动 CPU 卸载
- **LoRA 合成**：高效的 LoRA 堆叠与合成
- **ComfyUI 集成**：与工作流无缝配合

## 安装

### 快速安装（v1.60 — 已简化）

**前置条件：**
- 必须已安装 ComfyUI-nunchaku

1. 将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git
```

2. 重启 ComfyUI

**就这么简单！** 节点会自动出现在 ComfyUI 的节点菜单中。

### 手动安装（备选）

若您更喜欢手动安装，或使用 macOS/Linux：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git
```

然后重启 ComfyUI。

## 系统要求

- Python 3.11+
- ComfyUI（建议使用最新版本）
- ComfyUI-nunchaku（必需）
- 支持 CUDA 的 GPU（可选，但建议用于性能）
