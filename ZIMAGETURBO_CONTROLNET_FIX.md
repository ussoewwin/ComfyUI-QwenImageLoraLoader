# ComfyUI-QwenImageLoraLoader側のZ-Image-Turbo ControlNet対応修正の詳細解説

## 追加修正が必要になった原因

非公式Z-Image-Turboローダー側（`ComfyUI-nunchaku-unofficial-z-image-turbo-loader`）で`NunchakuZImageTransformer2DModel.forward`をモンキーパッチし、`transformer_options`と`control`パラメータを受け付けるようにした。

しかし、ComfyUI-QwenImageLoraLoader側の`ComfyZImageTurboWrapper`クラスが、これらのパラメータを下位の`NunchakuZImageTransformer2DModel.forward`に正しく伝達していなかった。

`ComfyZImageTurboWrapper`はLoRA合成とフォワードパスを分離するラッパーで、`forward`メソッドで受け取った引数を`_execute_model`メソッド経由で`NunchakuZImageTransformer2DModel.forward`に渡す。この伝達経路に`transformer_options`と`control`を追加する必要があった。

## 修正したファイル

`ComfyUI-QwenImageLoraLoader/wrappers/zimageturbo.py`

## 修正内容

### 修正1: forwardメソッドのシグネチャ（76行目から85行目）

#### 修正前

```python
def forward(
        self,
        x,
        timestep,
        context=None,
        y=None,
        guidance=None,
        **kwargs,
):
```

#### 修正後

```python
def forward(
        self,
        x,
        timestep,
        context=None,
        y=None,
        guidance=None,
        control=None,
        transformer_options={},
        **kwargs,
):
```

#### 意味の説明

`forward`メソッドのシグネチャに`control`と`transformer_options`を追加。ComfyUIのModelPatcherからこれらを受け取れるようにした。

### 修正2: forwardメソッド内のtransformer_optionsマージ処理（93行目から100行目）

#### 修正内容

```python
# Remove guidance, transformer_options, and attention_mask from kwargs
if "guidance" in kwargs:
    kwargs.pop("guidance")
if "transformer_options" in kwargs:
    if isinstance(transformer_options, dict) and isinstance(kwargs["transformer_options"], dict):
        transformer_options = {**transformer_options, **kwargs.pop("transformer_options")}
    else:
        kwargs.pop("transformer_options")
if "attention_mask" in kwargs:
    kwargs.pop("attention_mask")
```

#### 意味の説明

`kwargs`に`transformer_options`が含まれている場合、明示的なパラメータとマージ。明示的なパラメータが優先。

### 修正3: _execute_modelメソッドへの引数伝達（219行目、221行目）

#### 修正内容

`forward`メソッド内で`_execute_model`を呼び出す際に、`control`と`transformer_options`を渡すように変更：

```python
with cache_context(self._cache_context):
    out = self._execute_model(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
else:
    out = self._execute_model(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
```

#### 意味の説明

`_execute_model`に`control`と`transformer_options`を渡すことで、下位の`NunchakuZImageTransformer2DModel.forward`まで伝達できるようにした。

### 修正4: _execute_modelメソッドのシグネチャ（239行目）

#### 修正前

```python
def _execute_model(self, x, timestep, context, y, guidance, **kwargs):
```

#### 修正後

```python
def _execute_model(self, x, timestep, context, y, guidance, control, transformer_options, **kwargs):
```

#### 意味の説明

`_execute_model`のシグネチャに`control`と`transformer_options`を追加。

### 修正5: customized_forward使用時の伝達（282行目から300行目）

#### 修正内容

`customized_forward`が設定されている場合（非公式ZITローダー側のカスタムフォワードを使用する場合）の処理：

```python
if self.customized_forward:
    with torch.inference_mode():
        forward_kwargs_cleaned = {k: v for k, v in self.forward_kwargs.items() if k not in ("guidance", "ref_latents", "transformer_options", "attention_mask")}
        transformer_options_cleaned = dict(transformer_options) if transformer_options else {}
        transformer_options_cleaned.pop("guidance", None)
        transformer_options_cleaned.pop("ref_latents", None)
        
        return self.customized_forward(
            self.model,
            hidden_states=x,
            encoder_hidden_states=context,
            timestep=timestep,
            guidance=guidance if self.config.get("guidance_embed", False) else None,
            ref_latents=ref_latents_value,
            control=control,
            transformer_options=transformer_options_cleaned,
            **forward_kwargs_cleaned,
            **kwargs,
        )
```

#### 意味の説明

`customized_forward`呼び出し時に`control`と`transformer_options`を渡す。`transformer_options`から`guidance`と`ref_latents`を除去（重複を避けるため）。

### 修正6: 直接forward呼び出し時の伝達（389行目から409行目）

#### 修正内容

`customized_forward`が設定されていない場合（直接`NunchakuZImageTransformer2DModel.forward`を呼び出す場合）の処理：

```python
# Call Z-Image-Turbo forward with correct signature
# Pass control and transformer_options to allow Model Patcher (double_block patches) to work
# Check if the model's forward method accepts control parameter
import inspect
forward_sig = inspect.signature(self.model.forward)
forward_params = set(forward_sig.parameters.keys())

zimage_kwargs_clean = {k: v for k, v in zimage_kwargs.items() if k not in ('control', 'transformer_options')}

# Build kwargs based on what the forward method accepts
forward_kwargs = zimage_kwargs_clean.copy()
if 'control' in forward_params:
    forward_kwargs['control'] = control
if 'transformer_options' in forward_params:
    forward_kwargs['transformer_options'] = transformer_options

model_output = self.model(
    x_list,
    t_zimage,
    cap_feats=cap_feats,
    **forward_kwargs
)
```

#### 意味の説明

- `inspect.signature`で`self.model.forward`のシグネチャを確認
- `control`と`transformer_options`が受け付けられる場合のみ、それらを`forward_kwargs`に追加
- これにより、非公式ZITローダー側の`_patched_forward`（モンキーパッチされたforwardメソッド）が正しく`transformer_options`と`control`を受け取れる

## 重要な設計判断

### inspect.signatureによる動的パラメータチェック

`inspect.signature`を使って、`forward`メソッドが`control`と`transformer_options`を受け付けるかどうかを動的に確認する。これにより：

1. 後方互換性が保たれる（これらのパラメータを受け付けない旧バージョンでもエラーにならない）
2. 非公式ZITローダー側のモンキーパッチが適用されているかどうかに関係なく、適切に動作する
3. 将来的な拡張に対応しやすい

### 非公式ZITローダー側との連携

この修正により、以下の連携フローが実現される：

1. ComfyUIのModelPatcherが`transformer_options`と`control`を`ComfyZImageTurboWrapper.forward`に渡す
2. `ComfyZImageTurboWrapper.forward`がこれらを`_execute_model`に渡す
3. `_execute_model`が`inspect.signature`で確認してから`NunchakuZImageTransformer2DModel.forward`に渡す
4. 非公式ZITローダー側の`_patched_forward`（モンキーパッチされたforward）が`transformer_options`と`control`を受け取り、`double_block`パッチを適用する

## まとめ

ComfyUI-QwenImageLoraLoader側の`ComfyZImageTurboWrapper`クラスを修正し、`control`と`transformer_options`パラメータを正しく伝達するようにした。

主な変更点は：

1. `forward`メソッドのシグネチャに`control`と`transformer_options`を追加
2. `_execute_model`メソッドのシグネチャに`control`と`transformer_options`を追加
3. `inspect.signature`を使って、下位の`forward`メソッドがこれらのパラメータを受け付ける場合のみ渡すように実装

これにより、非公式ZITローダー側で追加したControlNet対応のモンキーパッチが正しく動作し、ControlNetとLoRAの両立が実現される。

