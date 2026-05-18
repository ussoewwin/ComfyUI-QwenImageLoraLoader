# ComfyUI-Nunchaku QwenImage＆ZImageTurboLoraStack

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../README.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

用于 Nunchaku Qwen Image 和 Z-ImageTurbo 模型的 ComfyUI 自定义节点，支持加载和应用 LoRA（低秩适配）。兼容 ComfyUI Nodes 2.0。**v4 功能需要 Nunchaku 1.2.0+ 和 ComfyUI-Nunchaku 1.2.0+。**

## ⚠️ **开发状态**

**目前正在开发和测试中。正在大量输出调试日志。这不影响功能。**

> 最新版本: [v2.4.4 on GitHub Releases](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.4)
>

## 来源

此 LoRA 加载器从 GavChap 的分支中提取并修改：
- **原始分支**: [GavChap/ComfyUI-nunchaku (qwen-lora-suport-standalone 分支)](https://github.com/GavChap/ComfyUI-nunchaku/tree/qwen-lora-suport-standalone)
- **提取**: 从完整分支中提取 LoRA 功能，创建独立的自定义节点
- **集成**: 修改为与官方 ComfyUI-nunchaku 插件一起工作

## 🎉 重大更新: v2.2.0 - 新增 Nunchaku Z Image Turbo LoRA 支持！

详细技术说明，请参阅 [v2.2.0 发行说明](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.0)

## 🎉 重大更新: v1.60 - 简化安装（无需集成！）

有关安装说明、功能和要求，请参阅 [安装指南](../md/installation.md)。

## v1.57 及更早版本用户的升级指南

如果您安装了 v1.57 或更早版本，并且在 ComfyUI-nunchaku 的 `__init__.py` 中有集成代码，请参阅 [UPGRADE_GUIDE_V1.57.md](../md/UPGRADE_GUIDE_V1.57.md) 了解详细的升级说明。

## 使用方法

### 调试日志（可选）

默认情况下，详细的调试日志是**静音的**。如果您需要详细的调试输出（密钥扩散/密钥映射检查、`[APPLY]`、`[AWQ_MOD]` 等），请在启动 ComfyUI 之前设置环境变量 `nunchaku_log=1`。

### 可用节点

- **NunchakuQwenImageLoraLoader**: 单 LoRA 加载器

<img src="images/single_loader.png" alt="NunchakuQwenImageLoraLoader: 单 LoRA 加载器" width="400">

- **NunchakuQwenImageLoraStack**: 多 LoRA 堆叠器，带动态 UI（旧版）

<img src="images/legacy_stack.png" alt="NunchakuQwenImageLoraStack: 多 LoRA 堆叠器，带动态 UI（旧版）" width="400">

- **NunchakuQwenImageLoraStackV1**: 多 LoRA 堆叠器，rgthree 风格 UI
  - 简洁、极简的设计，灵感来自 [Power Lora Loader (rgthree-comfy)](https://github.com/rgthree/rgthree-comfy)。每行包含切换、LoRA 名称和强度。
  - ⚠️ **注意**: **无法**与 ComfyUI Nodes 2.0 正常工作。请使用标准 (LiteGraph) 画布。
  - ⚠️ **与 ComfyUI Nodes 2.0 一起使用时，按 F5 刷新将反映更改。**

<img src="images/qiv1_stack.png" alt="NunchakuQwenImageLoraStackV1: 多 LoRA 堆叠器，rgthree 风格 UI" width="400">

- **NunchakuQwenImageLoraStackV2**: 多 LoRA 堆叠器，带动态 UI - 兼容 ComfyUI Nodes 2.0 (Beta)
  - **AWQ 调制层支持**: 包含通过 "Apply AWQ Mod" 切换将 LoRA 应用于 AWQ 量化调制层 (`img_mod.1` / `txt_mod.1`) 的实验性支持。此功能使用运行时猴子补丁和手动平面注入来修复噪声问题。⚠️ **警告**: **这是一个实验性功能，目前仅在 V2 节点中实现。** 如果没有发现问题，此功能也将应用于 V1 和 V3 节点。

<img src="images/qiv2_stack.png" alt="NunchakuQwenImageLoraStackV2: 多 LoRA 堆叠器，带动态 UI - 兼容 ComfyUI Nodes 2.0 (Beta)" width="400">

- **NunchakuQwenImageLoraStackV3**: 多 LoRA 堆叠器，带动态 UI - 兼容 ComfyUI Nodes 2.0 (Beta)
  - **AWQ 调制层支持**: AWQ 量化调制层 (`img_mod.1` / `txt_mod.1`) LoRA 应用**始终启用**（无需切换）。此功能使用运行时猴子补丁和手动平面注入来修复噪声问题。✅ **V3 节点默认始终应用 AWQ 调制层 LoRA。**

<img src="images/qiv3_stack.png" alt="NunchakuQwenImageLoraStackV3: 多 LoRA 堆叠器，带动态 UI - 兼容 ComfyUI Nodes 2.0 (Beta)" width="400">

- **NunchakuZImageTurboLoraStackV1**: Z-Image-Turbo LoRA 堆叠器，rgthree 风格 UI
  - 简洁、极简的设计，灵感来自 [Power Lora Loader (rgthree-comfy)](https://github.com/rgthree/rgthree-comfy)。每行包含切换、LoRA 名称和强度。仅用于官方 Nunchaku Z-Image 加载器。使用 compose_loras_v2。
  - ⚠️ **注意**: **无法**与 ComfyUI Nodes 2.0 正常工作。请使用标准 (LiteGraph) 画布。
  - ⚠️ **与 ComfyUI Nodes 2.0 一起使用时，按 F5 刷新将反映更改。**

<img src="images/zitlorav1.png" alt="NunchakuZImageTurboLoraStackV1: Z-Image-Turbo LoRA 堆叠器，rgthree 风格 UI" width="400">

- **NunchakuZImageTurboLoraStackV4**: Z-Image-Turbo LoRA 堆叠器，带动态 UI - 标准 ComfyUI LoRA 加载器格式 (CLIP 输入/输出) - 兼容 ComfyUI Nodes 2.0

<img src="images/zitv4_stack.png" alt="NunchakuZImageTurboLoraStackV4: Z-Image-Turbo LoRA 堆叠器，带动态 UI - 标准 ComfyUI LoRA 加载器格式 (CLIP 输入/输出) - 兼容 ComfyUI Nodes 2.0" width="400">

- **NunchakuZImageTurboLoraStackV2**: Z-Image-Turbo LoRA 堆叠器，带动态 UI - **仅非官方加载器** - 兼容 ComfyUI Nodes 2.0
  - ⚠️ **警告**: 此节点**仅兼容**来自 [ComfyUI-nunchaku-unofficial-loader](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader) 的非官方 Nunchaku Z-Image-Turbo DiT 加载器
  - ⚠️ **不兼容**: 此节点**不兼容**来自 ComfyUI-Nunchaku 的官方 Nunchaku Z-Image-Turbo DiT 加载器

### 基本用法

**对于 Nunchaku Qwen Image 模型：**
1. 使用 `Nunchaku Qwen Image DiT Loader` 加载您的 Nunchaku Qwen Image 模型
2. 添加 `NunchakuQwenImageLoraLoader` 或 `NunchakuQwenImageLoraStack` 节点
3. 选择您的 LoRA 文件并设置强度
4. 连接到您的工作流

**对于 Nunchaku Z-Image-Turbo 模型：**
1. 使用 `Nunchaku Z-Image DiT Loader` (官方 ComfyUI-Nunchaku) 加载您的 Nunchaku Z-Image-Turbo 模型
2. 添加 `Nunchaku Z-Image-Turbo LoRA Stack V4` 节点
3. 连接 CLIP 输入 (v4 中必需)
4. 选择您的 LoRA 文件并设置强度
5. 连接到您的工作流

## 功能

- **易于安装**: 简单的 git clone 安装
- **独立运行**: 无需集成代码 (v1.60+)
- **自动节点发现**: ComfyUI 自动加载自定义节点
- **错误处理**: 全面的错误检查和用户反馈
- **问题 #1 已修复**: 通过改进的路径检测解决了 [ComfyUI\custom_nodes not found 错误](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/1)（感谢 @mcv1234 的解决方案）
- **问题 #2 已修复**: 通过使用专用 Python 脚本进行正确的 UTF-8 编码，修复了导致 `SyntaxError: (unicode error)` 的 UTF-8 编码错误（感谢 @AHEKOT 的错误报告）
- **问题 #3 已修复 (v1.4.0)**: 通过实现带有基于哈希的更改检测的正确 IS_CHANGED 方法，解决了 [节点中断缓存进度错误](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/3)（感谢 @AHEKOT 的错误报告）
- **问题 #10 已修复**: 添加了便携式 ComfyUI 支持，包括嵌入式 Python 检测 ([问题 #10](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/10)) - **特别感谢**: 这一关键功能是由 @vvhitevvizard 提出的，他识别了便携式 ComfyUI 安装中对嵌入式 Python 支持的需求。如果没有这个建议，便携式 ComfyUI 用户将无法使用此 LoRA 加载器。
- **PR #48 已合并 (v2.3.8)**: 改进了 PEFT 格式 LoRA 检测并添加了安全跳过日志 ([PR #48](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/48)) - **特别感谢**: 我们非常感谢 [@avan06](https://github.com/avan06) 识别并修复了一个关键的映射缺陷，该缺陷导致 PEFT 格式 LoRA 被错误地跳过。此修复显著提高了 LoRA 格式兼容性，并确保正确检测使用 Hugging Face PEFT 库创建的 LoRA 文件。
- **PR #49 已合并 (v2.4.0)**: 添加了 Nunchaku Qwen Image LoRA Stack V1，rgthree 风格 UI ([PR #49](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/49)) - **特别感谢**: 我们非常感谢 [@avan06](https://github.com/avan06) 提出了受 Power Lora Loader (rgthree-comfy) 启发的简洁、极简界面。此贡献带来了优雅的 LoRA 行布局，包含切换、LoRA 名称和强度控制。我们长期以来一直感觉到需要这样的 UI，但由于技术能力无法实现；此 PR 满足了这一需求。

## 要求

- ComfyUI
- ComfyUI-nunchaku 插件 (官方版本，无需修改)
- PyTorch
- Python 3.11+

## 兼容性

此节点设计用于：
- ComfyUI-nunchaku 插件 (官方版本)
- Nunchaku Qwen Image 模型
- 标准 ComfyUI 工作流

## 故障排除

### 错误: "ModuleNotFoundError: No module named 'nunchaku'"

**问题**: nunchaku 包未安装。

**解决方案**:
1. 从官方仓库安装 ComfyUI-nunchaku 插件
2. 按照 nunchaku 安装说明安装 nunchaku wheel
3. 重启 ComfyUI

### 问题 #25: ComfyUI 0.4.0+ 模型管理错误
- **状态**: ⚠️ **取决于环境** - 可能需要 ComfyUI 核心修复

详细信息，请参阅 [COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md](../md/COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md)。

- **相关问题**:
  - [问题 #25](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/25) - `AttributeError: 'NunchakuModelPatcher' object has no attribute 'pinned'` 和 `model_config` 的深拷贝错误
  - [问题 #33](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/33) - `to_safely` 方法中的 `AttributeError: 'NoneType' object has no attribute 'to'` (已在 v2.1.0 中修复)
  - [ComfyUI 问题 #6590](https://github.com/comfyanonymous/ComfyUI/issues/6590) - `'NoneType' object has no attribute 'shape'`
  - [ComfyUI 问题 #6600](https://github.com/comfyanonymous/ComfyUI/issues/6600) - `'NoneType' object is not callable` (与加载器相关)
  - [ComfyUI 问题 #6532](https://github.com/comfyanonymous/ComfyUI/issues/6532) - 模型卸载后引用模型导致崩溃

### 错误: "No module named 'diffusers.models.transformers.transformer_z_image'"

**问题**: 使用 Qwen-Edit 模型或 Z-Image-Turbo 模型时，`diffusers` 库版本过旧会出现此错误。

**错误信息**: `ModuleNotFoundError: No module named 'diffusers.models.transformers.transformer_z_image'`

**根本原因**: 最可能的原因是 `diffusers` 库版本太旧，不包含 Z-Image-Turbo 模型支持所需的 `transformer_z_image` 模块。当 ComfyUI-nunchaku 的模型加载器尝试加载 Z-Image-Turbo 模型（或可能被检测为 Z-Image 格式的 Qwen-Edit 模型）时，它会尝试导入此模块，但它在旧版本的 `diffusers` 中不存在。此模块是在更高版本的 `diffusers` 中添加的，以支持 Z-Image-Turbo 模型。

**解决方案**: 将 `diffusers` 库更新到最新版本：

**如果使用虚拟环境 (venv)：**
```bash
pip install --upgrade diffusers
```

**如果使用 ComfyUI 的嵌入式 Python：**
```bash
ComfyUI\python_embeded\python.exe -m pip install --upgrade diffusers
```

**如何验证**: 更新后，重启 ComfyUI 并再次尝试加载您的模型。错误应该已解决。

**相关问题**: [问题 #38](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/38), [问题 #40](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/40)

## 已知限制

### LoKR (Lycoris) LoRA 支持
- **状态**: ❌ **不支持**
- **问题**: **不支持** Lycoris 创建的 LoKR 格式 LoRA。
  - **重要说明**: 此限制特别适用于 **Nunchaku 量化模型**。LoKR 格式 LoRA 可能适用于标准（非量化）Qwen Image 模型，但此节点专为 Nunchaku 模型设计。
  - ⚠️ **开发历史**: **我们花费了大量时间分析 LoKR 格式的内部结构并进行广泛的映射测试。尽管付出了这些努力，我们仍未找到将 LoKR 权重成功应用于 Nunchaku 量化模型的方法。** 实验性转换代码已测试，但由于不兼容问题最终被禁用。
  - 检测到 LoKR 权重时会自动跳过（实验性转换代码已禁用）。
  - 使用 SVD 近似（通过外部工具或脚本）转换为标准 LoRA 也已测试，发现应用于 Nunchaku 量化模型时**会产生噪声/伪影**。
- **结论**: 目前，我们尚未找到将 LoKR 权重成功应用于 Nunchaku 模型的方法。请使用标准 LoRA 格式。
- **支持的格式**:
  - ✅ **标准 LoRA (秩分解)**:
    - 支持的权重键：
      - `lora_up.weight` / `lora_down.weight`
      - `lora.up.weight` / `lora.down.weight`
      - `lora_A.weight` / `lora_B.weight`
      - `lora.A.weight` / `lora.B.weight`
    - 这些是 Kohya-ss、Diffusers 和大多数训练脚本生成的标准格式。
  - ✅ **PEFT 格式 LoRA** (v2.3.8+):
    - 支持的权重键（示例）：
      - `lora_A.default.weight` / `lora_B.default.weight`
      - `lora.up.default.weight` / `lora.down.default.weight`
    - 支持 `lora_A`/`lora_B` 和 `.weight` 之间的附加标签（例如 `.default`）。
    - 这些格式由 Hugging Face PEFT 库生成。v2.3.8 (PR #48) 中添加了对它们的支持。
  - ❌ **LoKR (Lycoris)**: 不支持 (键如 `lokr_w1`, `lokr_w2`)
  - ❌ **LoHa**: 不支持 (键如 `hada_w1`, `hada_w2`)
  - ❌ **IA3**: 不支持
- **相关问题**:
  - [问题 #29](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/29) - LyCORIS / LoKr Qwen Image LoRA 未被 ComfyUI 识别
  - [问题 #44](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/44) - [MISS] 找到模块但不支持/缺少 proj_down/proj_up (性能缓慢问题已在 v2.2.8 中修复)

### context_refiner 和 noise_refiner 层支持
- **状态**: ❌ **当前不支持**
- **问题**: **当前不支持** `context_refiner` 和 `noise_refiner` 层的 LoRA 键。这些是 Z-Image-Turbo 模型中用于细化功能的专用层。
- **详细信息**:
  - ⚠️ **测试限制**: 我们尚未使用实际模型测试包含 `context_refiner` 和 `noise_refiner` 层的 LoRA。我们无法测试它们的原因是 **我们不知道哪些 LoRA 是这种格式**。如果我们能识别出包含这些层的特定 LoRA，我们就可以进行测试。
  - 在没有测试的情况下，我们无法确定这些层的正确密钥映射是否可行。当前的密钥映射系统不包含这些层的映射，它们**目前无法工作**。
- **结论**: 由于我们未能测试包含这些层的 LoRA，因此未实现对 `context_refiner` 和 `noise_refiner` 层的支持。即使我们测试了它们，也无法确定是否可以确定正确的密钥映射。请仅使用针对标准转换器层的 LoRA。
- **相关问题**:
  - [#41](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/41) - 无法加载context_refiner和noise_refiner层的参数

### RES4LYF Sampler 兼容性问题
- **状态**: ✅ 已在 ComfyUI-nunchaku v1.0.2 中修复
- **问题**: 将 RES4LYF sampler 与 LoRA 一起使用时发生设备不匹配错误 ([问题 #7](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/7), [问题 #8](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/8))
- **修复**: 该问题已在 [ComfyUI-nunchaku v1.0.2](https://github.com/nunchaku-tech/ComfyUI-nunchaku/releases/tag/v1.0.2) 中由 @devgdovg 在 PR #600 中修复。此修复是在 ComfyUI-nunchaku 的代码库中实现的，而不是在此 LoRA 加载器中。
- **要求**: 更新到 ComfyUI-nunchaku v1.0.2 或更高版本，以将 RES4LYF sampler 与 LoRA 一起使用
- **相关问题**:
  - [问题 #7](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/7) - RES4LYF sampler 设备不匹配错误
  - [问题 #8](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/8) - RES4LYF sampler 兼容性问题

## 更新日志

### v2.4.5 (最新)
- **已添加**: 在 `zhmd/` 目录下新增中文文档页面（README 与发行说明），并在英文与中文 README 及发行说明页面提供双语语言切换。

### v2.4.4
- **已修复**: 在 `compose_loras_v2` 中恢复 v2.2.7 的首个 LoRA 重复文件读取消除功能（v2.3.0 AWQ 重构中的回归）。首个 LoRA 只加载一次并在主循环中重用，为 Qwen Image 和 Z-Image-Turbo 加载器减少了 50% 的重复文件 I/O、反序列化和密钥分类。与 AWQ 调制层猴子补丁完全兼容。
- **技术详情**: 请参阅 [v2.4.4 发行说明](v2.4.4.md) 获取完整说明

### v2.4.3
- **已修复**: 修复了 Z-Image / SVDQ 与 ComfyUI 延迟（惰性）`Linear` 权重的崩溃问题（`AttributeError: 'NoneType' object has no attribute 'dtype'`），通过修补 `SVDQW4A4Linear.from_linear` 和 `fuse_to_svdquant_linear`，包括加载顺序差异的启动重试。
- **技术详情**: 请参阅 [v2.4.3 发行说明](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.3) 获取完整说明

### v2.4.2
- **已修复**: 修复了 Nunchaku Qwen Image 模型的 Qwen Image ControlNet（例如 Fun ControlNet）问题 — `ComfyQwenImageWrapper` 现在暴露 `process_img` 并将 ControlNet 所需的属性（`patch_size`, `pe_embedder`, `img_in`, `txt_norm`, `txt_in`, `time_text_embed`）转发到内部模型，以便当基础模型是包装器时 Union ControlNet 可以工作。
- **已修复**: 修复了访问 `model_wrapper.model` 时的 RecursionError（例如在 NunchakuQwenImageLoraStackV3 中）— `__getattr__` 现在通过 `_modules` 而不是 `self.model` 获取内部模型，以避免无限递归。
- **技术详情**: 请参阅 [v2.4.2 发行说明](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.2) 获取完整说明

### v2.4.1
- **已添加**: Nunchaku Z-Image-Turbo LoRA Stack V1，rgthree 风格 UI - 与 Qwen Image LoRA Stack V1 相同的布局：每行包含切换、LoRA 名称和强度。仅用于官方 Nunchaku Z-Image 加载器。使用 compose_loras_v2。无法与 ComfyUI Nodes 2.0 正常工作；与 Nodes 2.0 一起使用时，按 F5 刷新将反映更改。
- **相关问题**: [问题 #12](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/12) - 请求更好的 LoRA 选项 (rgthree 风格 UI), [问题 #36](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/36) - 请求启用/禁用 LoRA 功能

### v2.4.0
- **已添加**: Nunchaku Qwen Image LoRA Stack V1，rgthree 风格 UI - 受 Power Lora Loader (rgthree-comfy) 启发的简洁、极简界面。每行包含切换、LoRA 名称和强度。
- **已合并**: [PR #49](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/49) - feat(qwen_lora): 添加 Nunchaku Qwen Image LoRA Stack V4，rgthree 风格 UI (由 [avan06](https://github.com/avan06) 提出)
- **注意**: 无法与 ComfyUI Nodes 2.0 正常工作。请使用标准 (LiteGraph) 画布。
- **相关问题**: [问题 #12](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/12) - 请求更好的 LoRA 选项 (rgthree 风格 UI), [问题 #36](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/36) - 请求启用/禁用 LoRA 功能
- **技术详情**: 请参阅 [v2.4.0 发行说明](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.0) 获取完整说明

### 先前版本 (v2.3.0 到 v2.3.9)

有关 v2.3.0 到 v2.3.9 的详细发行说明，请参阅 [RELEASE_NOTES_V2.3.0_TO_V2.3.9.md](../RELEASE_NOTES/RELEASE_NOTES_V2.3.0_TO_V2.3.9.md)。

此文档包含有关这些版本的所有错误修复、功能和技术细节的全面信息。

### 先前版本 (v1.6.0 到 v2.2.8)

有关 v1.6.0 到 v2.2.8 的详细发行说明，请参阅 [RELEASE_NOTES_V1.6.0_TO_V2.2.8.md](../RELEASE_NOTES/RELEASE_NOTES_V1.6.0_TO_V2.2.8.md)。

此文档包含有关这些版本的所有错误修复、功能和技术细节的全面信息。

### 先前版本 (v1.0.0 到 v1.57)

有关 v1.0.0 到 v1.57 的详细发行说明，请参阅 [RELEASE_NOTES_V1.0.0_TO_V1.57.md](../RELEASE_NOTES/RELEASE_NOTES_V1.0.0_TO_V1.57.md)。

此文档包含有关这些版本的所有错误修复、功能和技术细节的全面信息。

## 许可证

本项目根据 MIT 许可证获得许可。
