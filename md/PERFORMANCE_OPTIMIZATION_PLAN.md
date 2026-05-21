# 高速化計画

`wrappers/qwenimage.py` と `nunchaku_code/lora_qwen.py` を実際に読んだ上で作成。推測は入れない。

---

## 1. `_load_lora_state_dict_robust` にファイルキャッシュを追加

### 現状の事実

```python
def _load_lora_state_dict_robust(path_or_dict: Union[str, Dict]) -> Dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    path = str(path_or_dict)
    if os.path.exists(path):
        return comfy.utils.load_torch_file(path)
```

`compose_loras_v2` のループ内で各 `lora_configs` 要素ごとに呼ばれる。同一ファイルパスが複数回含まれていても、毎回 `comfy.utils.load_torch_file(path)` を呼ぶ。辞書キャッシュは存在しない。

### 高速化案

`_load_lora_state_dict_robust` に `functools.lru_cache(maxsize=None)` を付与し、同一パスの2回目以降のファイル読み込みを省略する。

```python
@functools.lru_cache(maxsize=None)
def _load_lora_state_dict_robust(path_or_dict: Union[str, Dict]) -> Dict:
    ...
```

`path_or_dict` が `dict` の場合はキャッシュ不要だが、文字列/Path の場合はキャッシュが有効。`lru_cache` は `hashable` な引数のみ受け付けるため、文字列パスに限定する。

### 影響範囲

- `compose_loras_v2` 内の `for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):` ループ
- 同一 LoRA ファイルが複数回指定された場合、2回目以降はメモリから取得

---

## 2. デバイス転送のバッチ化

### 現状の事実

```python
all_A.append(A.to(dtype=target_dtype, device=target_device))
all_B_scaled.append((B * scale).to(dtype=target_dtype, device=target_device))
```

各 LoRA の各 weight（A, B）ごとに個別の `.to(device=...)` を発行。`torch.cat` は転送後に行われる。

### 高速化案

CPU 上で `torch.cat` または `torch.stack` してから、1回の `.to(device)` を発行する。

```python
# 現状
all_A.append(A.to(dtype=target_dtype, device=target_device))
# ...
final_A = torch.cat(all_A, dim=0)

# 変更後
all_A_cpu.append(A.to(dtype=target_dtype, device="cpu"))
# ...
final_A_cpu = torch.cat(all_A_cpu, dim=0)
final_A = final_A_cpu.to(device=target_device)
```

### ただし

- `target_device` は `module.qweight.device` または `module.proj_down.device` で、モジュールごとに異なる可能性がある
- 同一モジュール内の `all_A` は同一デバイスに転送されるため、モジュール単位でバッチ化は有効

---

## 3. `_awq_lora_forward` 内の `.to()` 削除または事前転送

### 現状の事実

```python
for (A_local, B_local) in module._nunchaku_lora_bundle:
    A_local = A_local.to(device=device, dtype=dtype)
    B_local = B_local.to(device=device, dtype=dtype)
    lora_mid = x_flat @ A_local.transpose(0, 1)
    lora_term = lora_mid @ B_local.transpose(0, 1)
```

`_apply_lora_to_module` で `A_local`, `B_local` をモジュールにアタッチする際、既に `target_device` に転送している：

```python
all_A.append(A.to(dtype=target_dtype, device=target_device))
# ...
mod._nunchaku_lora_bundle.append((A, B))
```

しかし `_awq_lora_forward` では毎回 `.to(device=device, dtype=dtype)` を発行しており、すでに同じデバイス/dtype にあるテンソルに対する無駄なコピーが発生している可能性がある。

### 高速化案

`_nunchaku_lora_bundle` に格納する時点で、既に `target_device` / `target_dtype` に変換済みのテンソルを格納する。`_awq_lora_forward` 内の `.to()` を削除または、デバイス/dtype が一致している場合はスキップする。

```python
# _awq_lora_forward 内
for (A_local, B_local) in module._nunchaku_lora_bundle:
    # 既に同じデバイス/dtypeならスキップ
    if A_local.device != device or A_local.dtype != dtype:
        A_local = A_local.to(device=device, dtype=dtype)
    if B_local.device != device or B_local.dtype != dtype:
        B_local = B_local.to(device=device, dtype=dtype)
    ...
```

---

## 4. `_get_module_by_name` のパスキャッシュ

### 現状の事実

```python
def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    module = model
    for part in name.split("."):
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and _is_indexable_module(module):
            module = module[int(part)]
        else:
            return None
    return module
```

`compose_loras_v2` の `for module_name_key, weight_list in aggregated_weights.items():` ループ内で、毎回 `_resolve_module_name` → `_get_module_by_name` を呼ぶ。同一 `module_name_key` が複数回現れる場合、毎回ツリーを走査する。

### 高速化案

`compose_loras_v2` 内にローカルキャッシュ `{name: module}` を設け、2回目以降はキャッシュを使用する。

```python
_module_cache = {}
# ...
for module_name_key, weight_list in aggregated_weights.items():
    if module_name_key in _module_cache:
        resolved_name, module = _module_cache[module_name_key]
    else:
        resolved_name, module = _resolve_module_name(model, module_name_key)
        _module_cache[module_name_key] = (resolved_name, module)
```

---

## 5. `all_A` / `all_B_scaled` の事前確保

### 現状の事実

```python
all_A = []
all_B_scaled = []
for w in weight_list:
    # ...
    all_A.append(A.to(...))
    all_B_scaled.append((B * scale).to(...))
final_A = torch.cat(all_A, dim=0)
final_B = torch.cat(all_B_scaled, dim=1)
```

`append` 後に `torch.cat` を行う。`cat` は新しいテンソルを割り当てるため、`append` された各テンソルのメモリは断片化する。

### 高速化案

`weight_list` のサイズが事前にわかっているため、`torch.empty` で事前に確保し、`copy_` で埋める。ただし、各 A/B のサイズ（rank）が異なるため、連続したメモリ確保は困難。`torch.cat` の代替として `torch.zeros` で事前確保する方法もあるが、複雑さが増す。

---

## 結論

| 最適化 | 期待効果 | 実装難易度 |
|--------|----------|------------|
| `_load_lora_state_dict_robust` の `lru_cache` | 中（同一ファイルの2回目以降省略） | 低 |
| デバイス転送のバッチ化（モジュール単位） | 小〜中（`.to()` の回数削減） | 中 |
| `_awq_lora_forward` の `.to()` 条件スキップ | 小（無駄なコピー排除） | 低 |
| `_get_module_by_name` のローカルキャッシュ | 小（走査コスト削減） | 低 |
| `all_A` / `all_B_scaled` の事前確保 | 小（メモリ断片化軽減） | 高（rank 可変のため） |

**最も効果が高く実装が容易なのは 1（ファイルキャッシュ）と 3（AWQ forward の `.to()` スキップ）と 4（モジュールパスキャッシュ）です。**
