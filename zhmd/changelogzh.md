<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/RELEASE_NOTES/changelog.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

### v2.6.0 (最新)
- **已更改**: ControlNet 节点名称由 `NunchakuQI&ZITDiffsynthControlnet` 改为 **`Nunchaku ZI Diffsynth Controlnet&Krea2 LoRA ControlNet`**（内部类名不变：`NunchakuQwenImageDiffsynthControlnet`，已有工作流无需修改即可加载）。
- **明确支持范围**:
  - **Z-Image-Turbo（ZI 路由）**：支持 **Nunchaku**（量化）以及所有非量化 / 量化 Z-Image 模型。
  - **Qwen Image（QI 路由）**：支持所有**非 Nunchaku** 的 Qwen Image 模型（非量化与量化均支持，如 HSWQ ConvRot INT8）。**Nunchaku Qwen Image 不受支持**（量化后的 hidden-state 尺度不匹配会导致输出损坏）。QI 路由请使用标准 bf16 Qwen Image 模型，ZI 路由请使用 Nunchaku Z-Image-Turbo。
- **已修复**: 路由分类现在能识别标准（非 Nunchaku）`QwenImageTransformer2DModel` 并路由到 `qwenimage_standard` 路径，DiffSynth ControlNet 可在标准 bf16 Qwen Image 模型上正常工作。
- **文档**: 更新 README（EN + ZH）— 节点图片、精确的支持范围说明与用法说明。
- **技术详情**: 参见 [v2.6.0 发行说明](v2.6.0.md) 获取完整说明

### v2.5.9
- **已修复**: ComfyUI 启动时来自 Hugging Face `transformers` `@auto_docstring` 的 `[ERROR]` 日志噪音 —— `DeepseekVLHybridImageProcessorKwargs`（`high_res_size`）、`Kimi_K25ImageProcessorKwargs`（`merge_size`）、`PaddleOCRVLImageProcessorKwargs`（`min_pixels` / `max_pixels`）共 13 行 "but not documented" 报错。上游 TypedDict docstring 无法通过校验：`high_res_size` 行多了一个前导空格导致解析器匹配不到；kimi 的 docstring 写的是 `merge_kernel_size` 而注解为 `merge_size`；paddleocr 完全未记录 `min_pixels` / `max_pixels`。
- **非侵入式修复**: 不修改 `site-packages`。`prestartup_script.py` 在进程内包装 `get_args_doc_from_source`（与 v2.4.7 CausalLM docstring 补丁同一设计），使这四个字段通过回退源字典被解析。
- **上游自动禁用（全自动）**: 每次 ComfyUI 启动都会探测上游；一旦 `transformers` 修复了 docstring，补丁会自动跳过自身。无需环境变量或用户开关。
- **文档**: 新增英文技术说明 `md/TRANSFORMERS_IMAGE_PROCESSOR_KWARGS_DOCSTRING_PATCH.md`。
- **技术详情**: 参见 [v2.5.9 发行说明](v2.5.9.md) 获取完整说明

### v2.5.8
- **已修复**: 当预编译 LoRA 缓存文件在磁盘上损坏（头部可读、数据区不可读）时，ComfyUI 不再因 Windows 致命 "page error" 崩溃。`load_precompiled()` 现在改为以普通字节方式读取缓存，而非使用基于 mmap 的 `load_file`，因此同一损坏会以普通 Python `OSError` 形式暴露，加载器会自动回退到完整的重新融合。
- **技术详情**: 参见 [v2.5.8 发行说明](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.5.8) 获取完整说明

### v2.5.7
- **已移除**: 从注册节点映射及文档中移除了旧版节点 `NunchakuQwenImageLoraStack`（"Nunchaku Qwen Image LoRA Stack (Legacy)"）。请使用 `NunchakuQwenImageLoraStackV1`、`V2` 或 `V3` 替代。
- **技术详情**: 请参阅 [v2.5.7 发行说明](v2.5.7.md) 获取完整说明

### v2.5.6
- **已添加**: AMD/ROCm 兼容性 — nunchaku 仅支持 NVIDIA CUDA，在 AMD/ROCm 系统上不可用。本包现在会在启动时探测 nunchaku 可用性（`_NUNCHAKU_AVAILABLE`），并在这类系统上自动禁用所有依赖 nunchaku 的节点（`NunchakuQwenImageLoraLoader/Stack`、`V1`/`V2`/`V3`、`NunchakuZImageTurboLoraStackV1`/`V4`）；它们将不再注册。
- **AMD/ROCm 上仍可使用**: 不依赖 nunchaku 的 Krea2 ControlNet 节点保持完整功能 — `Krea2ControlNetLoraLoader` 以及 `NunchakuQI&ZITDiffsynthControlnet` 的 Krea2 路由。
- **清晰的错误**: 运行时 nunchaku 导入已加防护，旧工作流会收到清晰的 `RuntimeError`（"nunchaku is required..."），而非曲折的 `ImportError`。缺少依赖时，启动时会输出 `[ROCm/AMD] nunchaku is not available...` 警告日志。
- **文档**: README（EN + ZH）已更新，新增 AMD / ROCm 兼容性章节。
- **技术详情**: 参见 [v2.5.6 发行说明](v2.5.6.md) 获取完整说明

### v2.5.5
- **已添加**: 在 `NunchakuQI&ZITDiffsynthControlnet` 中新增 Krea2 **openpose** ControlNet LoRA 支持（`krea2_turbo_openpose_controlnet.safetensors`）。
- **互斥路由**: Krea2 路由现在会根据 LoRA 文件自动检测控制子类型 —— **depth**（`first.weight` 扩展投影）与 **openpose**（纯块 LoRA）。openpose 分支与既有的 depth / Qwen Image / Z-Image / Nunchaku 路由完全隔离；现有行为不受任何影响。
- **原生参考潜变量条件**: 姿态图经 VAE 编码后，以原生 Krea2 参考潜变量（`index_timestep_zero`）注入 —— 这正是该 openpose 控制 LoRA 训练时所采用的条件通路（256 个块补丁，rank 32，作用于全部 28 个 DiT 块及 txtfusion 的 layerwise/refiner 块）。
- **新增 `use_kv_cache` 选项**（默认 `True`）: 隔离参考 K/V 模式 —— 在 t=0 时执行一次参考前向，预先计算每个块的 K/V，并在每个采样步作为额外 attention key 注入。与参考工作流使用的 ai-toolkit kv_cache 训练约定一致。设为 `False` 可回退到原生 joint 模式参考。
- **文档**: 英文 README 与中文 README 已更新，新增独立的 Krea2 Depth 控制 与 Krea2 OpenPose 控制 章节。
- **技术详情**: 参见 [v2.5.5 发行说明](v2.5.5.md) 获取完整说明

### v2.5.3
- **已变更**: 部分采用 PR #52 合并方案。
- **技术详情**: 请参阅 [v2.5.3 发行说明](v2.5.3.md) 获取完整说明

### v2.5.2
- **已修复**: 抑制了 ComfyUI 启动时的无害警告 `WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.`（[Issue #53](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/53)）。该警告出现，是因为必需的 `apply_rotary_emb` prestartup 垫片必须导入在模块层级导入 `torch` 的 `comfy.ldm` 模块（此时所有 CUDA 环境设置均已完成，因此无害）。在 `prestartup_script.py` 早期安装的一次性根日志过滤器仅丢弃这一条消息，其余所有日志照常输出。可通过 `QWENIMAGE_SUPPRESS_TORCH_WARNING=0` 关闭该抑制。
- **技术详情**: 请参阅 [v2.5.2 发行说明](v2.5.2.md) 获取完整说明

### v2.5.1
- **已添加**: 通过 `Krea2ControlNetLoraLoader` 以及 `NunchakuQI&ZITDiffsynthControlnet` 中的 Krea2 路由，新增 Krea2 depth ControlNet LoRA 支持。从 `controlnet` 文件夹加载 Krea2 depth controlnet-lora 文件（例如 `krea2-depth-control-lora.safetensors`），将其 `MODEL_PATCH` 输出连接到 controlnet 节点的 `model_patch` 输入，并在 Krea2 / SingleStreamDiT 模型上应用 depth 条件。
- **技术详情**: 请参阅 [v2.5.1 发行说明](v2.5.1.md) 获取完整说明

### v2.5.0
- **已添加**: 恢复了以前未注册的 NunchakuQI&ZITDiffsynthControlnet 节点。
- **已修复**: 修复了由于自定义前向循环跳过动态调整大小而导致 ControlNet 无法应用于 Nunchaku Qwen Image 模型的问题。
- **技术详情**: 请参阅 [v2.5.0 发行说明](v2.5.0.md) 获取完整说明

### v2.4.7
- **已修复**: ComfyUI 启动时，导入 Qwen3 VL / Qwen2.5 VL *CausalLMOutputWithPast 会触发 Hugging Face transformers 的 @auto_docstring，在控制台输出 [ERROR] loss / [ERROR] logits。这不是本节点 LoRA 加载逻辑的缺陷。由于上游何时修复尚不确定，本节点仅在 prestartup_script.py 内包装 get_args_doc_from_source 来吸收该问题（不修改 site-packages，不过滤 stderr）。
- **上游自动禁用（全自动）**: 每次 ComfyUI 启动时，补丁会探测上游 ModelOutputArgs 并运行子进程 Qwen VL 导入测试。一旦上游 transformers 修复，下次启动时将自动跳过补丁。无需环境变量或用户开关（与仍允许通过 QWENIMAGE_ROTARY_COMPAT 退出的 v2.4.6 apply_rotary_emb 兼容垫片不同）。
- **说明**: LoRA 行为未变。根因是导入这些 ModelOutput 类时（常由其他自定义节点或工作流触发）上游 transformers 对 Qwen VL @auto_docstring 的校验。
- **技术详情**: 请参阅 [v2.4.7 发行说明](v2.4.7.md) 获取完整说明

### v2.4.6
- **已修复**: ComfyUI 0.24.x 启动时 ComfyUI-nunchaku 导入 Qwen Image 节点失败的问题（ImportError: cannot import name 'apply_rotary_emb' from 'comfy.ldm.qwen_image.model'）。通过本自定义节点的早期 prestartup_script.py 垫片，将 apply_rotary_emb 别名为 ComfyUI 的 apply_rope1（无需修改 ComfyUI-nunchaku 文件）。
- **技术详情**: 请参阅 [v2.4.6 发行说明](v2.4.6.md) 获取完整说明

### v2.4.5
- **已添加**: 在 zhmd/ 目录下新增中文文档页面（README 与发行说明），并在英文与中文 README 及发行说明页面提供双语语言切换。
- **技术详情**: 请参阅 [v2.4.5 发行说明](v2.4.5.md) 获取完整说明

### v2.4.4
- **已修复**: 在 compose_loras_v2 中恢复 v2.2.7 的首个 LoRA 重复文件读取消除功能（v2.3.0 AWQ 重构中的回归）。首个 LoRA 只加载一次并在主循环中重用，为 Qwen Image 和 Z-Image-Turbo 加载器减少了 50% 的重复文件 I/O、反序列化和密钥分类。与 AWQ 调制层猴子补丁完全兼容。
- **技术详情**: 请参阅 [v2.4.4 发行说明](v2.4.4.md) 获取完整说明

### v2.4.3
- **已修复**: 修复了 Z-Image / SVDQ 与 ComfyUI 延迟（惰性）Linear 权重的崩溃问题（AttributeError: 'NoneType' object has no attribute 'dtype'），通过修补 SVDQW4A4Linear.from_linear 和 fuse_to_svdquant_linear，包括加载顺序差异的启动重试。
- **技术详情**: 请参阅 [v2.4.3 发行说明](v2.4.3.md) 获取完整说明

### v2.4.2
- **已修复**: 修复了 Nunchaku Qwen Image 模型的 Qwen Image ControlNet（例如 Fun ControlNet）问题 — ComfyQwenImageWrapper 现在暴露 process_img 并将 ControlNet 所需的属性（patch_size, pe_embedder, img_in, txt_norm, txt_in, time_text_embed）转发到内部模型，以便当基础模型是包装器时 Union ControlNet 可以工作。
- **已修复**: 修复了访问 model_wrapper.model 时的 RecursionError（例如在 NunchakuQwenImageLoraStackV3 中）— __getattr__ 现在通过 _modules 而不是 self.model 获取内部模型，以避免无限递归。
- **技术详情**: 请参阅 [v2.4.2 发行说明](v2.4.2.md) 获取完整说明

### v2.4.1
- **已添加**: Nunchaku Z-Image-Turbo LoRA Stack V1，rgthree 风格 UI - 与 Qwen Image LoRA Stack V1 相同的布局：每行包含切换、LoRA 名称和强度。仅用于官方 Nunchaku Z-Image 加载器。使用 compose_loras_v2。无法与 ComfyUI Nodes 2.0 正常工作；与 Nodes 2.0 一起使用时，按 F5 刷新将反映更改。
- **相关问题**: 问题 #12 - 请求更好的 LoRA 选项 (rgthree 风格 UI), 问题 #36 - 请求启用/禁用 LoRA 功能

### v2.4.0
- **已添加**: Nunchaku Qwen Image LoRA Stack V1，rgthree 风格 UI - 受 Power Lora Loader (rgthree-comfy) 启发的简洁、极简界面。每行包含切换、LoRA 名称和强度。
- **已合并**: PR #49 - feat(qwen_lora): 添加 Nunchaku Qwen Image LoRA Stack V4，rgthree 风格 UI (由 avan06 提出)
- **注意**: 无法与 ComfyUI Nodes 2.0 正常工作。请使用标准 (LiteGraph) 画布。
- **相关问题**: 问题 #12 - 请求更好的 LoRA 选项 (rgthree 风格 UI), 问题 #36 - 请求启用/禁用 LoRA 功能
- **技术详情**: 请参阅 [v2.4.0 发行说明](v2.4.0.md) 获取完整说明
