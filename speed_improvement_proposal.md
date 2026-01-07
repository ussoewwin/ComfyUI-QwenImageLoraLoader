# ComfyUI-QwenImageLoraLoader 速度改善提案書

**作成日**: 2026年1月7日  
**分析対象**: ComfyUI-QwenImageLoraLoader ノード全体

---

## 概要

本提案書は、ComfyUI-QwenImageLoraLoaderノードにおける潜在的な速度改善ポイントを分析し、具体的な改善案を提示するものである。

---

## 1. 現状分析

### 1.1 主要ファイル構成

| ファイル | 行数 | 役割 |
|---------|------|------|
| `nunchaku_code/lora_qwen.py` | 1521行 | LoRA合成・適用のコアロジック |
| `wrappers/qwenimage.py` | 387行 | ComfyUI用モデルラッパー |
| `wrappers/zimageturbo.py` | 478行 | Z-Image-Turbo用ラッパー |
| `nodes/lora/qwenimage.py` | 382行 | LoRAローダーノード定義 |

### 1.2 処理フロー

```
[LoRAローダーノード] 
    ↓
[モデルラッピング（初回のみ）]
    ↓
[forward()呼び出し]
    ↓
[LoRA変更検出]
    ↓
[compose_loras_v2() → LoRA合成]
    ↓
[モデル推論実行]
```

---

## 2. 速度ボトルネック分析

### 2.1 【最重要】 `lora_qwen.py` - LoRAファイルの重複読み込み

**場所**: `compose_loras_v2()` 関数（1191-1210行）

**問題点**:
```python
# 1回目: デバッグログ用（1193行目）
first_lora_state_dict = _load_lora_state_dict(first_lora_path_or_dict)
for key in first_lora_state_dict.keys():
    parsed_res = _classify_and_map_key(key)  # キー分類処理
    ...

# 2回目: 実際の処理用（1210行目）  
for lora_path_or_dict, strength in lora_configs:
    lora_state_dict = _load_lora_state_dict(lora_path_or_dict)  # ← 同じファイルを再読み込み！
    for key, value in lora_state_dict.items():
        parsed = _classify_and_map_key(key)  # ← キー分類処理も再実行！
```

> [!CAUTION]
> **同一LoRAファイルを2回読み込み、キー分類処理も2回実行している**

**改善案**:
1. **読み込み結果のキャッシュ再利用**
   - 1回目で読み込んだ`first_lora_state_dict`を2回目のループで再利用
   
2. **`_classify_and_map_key()`結果のメモ化**
   - `@functools.lru_cache`でキー分類結果をキャッシュ
   - 同じキーに対する重複計算を完全に排除

**推定改善効果**: LoRA読み込み・処理時間 **40-50%** 短縮

---

### 2.2 【高優先度】 `lora_qwen.py` - キーマッピング処理

**場所**: `_classify_and_map_key()` 関数（320-374行）

**問題点**: 
- すべてのLoRAキーに対して、KEY_MAPPING配列（約30個の正規表現パターン）を**順次走査**している
- safetensorsファイルには数百〜数千のキーが含まれるため、O(n×m)の計算量が発生

**改善案**: 
1. **プレフィックスベースのディスパッチ導入**
   - キーの先頭部分（`transformer_blocks`、`single_transformer_blocks`、`layers`など）でグループ分けし、該当グループのパターンのみ検索
   
2. **正規表現のプリコンパイルキャッシュ確認**
   - 現状: `re.compile()`済み ✅
   - 追加: `functools.lru_cache`デコレータで`_classify_and_map_key()`自体をキャッシュ可能

**推定改善効果**: LoRA読み込み時間 20-40% 短縮

---

### 2.3 【高優先度】 `lora_qwen.py` - LoRA状態辞書の読み込み

**場所**: `_load_lora_state_dict()` 関数（435-448行）

**問題点**:
```python
with safe_open(path, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)
```
- すべてのテンソルを**CPUメモリに一括読み込み**している
- 大容量LoRAファイル（100MB以上）で顕著なメモリ圧迫と読み込み遅延

**改善案**:
1. **遅延読み込み（Lazy Loading）の導入**
   - 必要なキーのみを逐次読み込む
   - 未使用キー（LoKR形式など）はスキップ

2. **メモリマップドI/O**
   - `mmap=True`オプションの検討（safetensorsサポート状況確認要）

**推定改善効果**: 大容量LoRA読み込み時間 30-50% 短縮、メモリ使用量削減

---

### 2.4 【中優先度】 `qwenimage.py` (wrapper) - LoRA変更検出

**場所**: `forward()` メソッド（128-142行）

**現状のコード**:
```python
loras_changed = False
if self._applied_loras is None or len(self._applied_loras) != len(self.loras):
    loras_changed = True
else:
    for applied, current in zip(self._applied_loras, self.loras):
        if applied != current:
            loras_changed = True
            break
```

**問題点**:
- 毎回のforward()でリスト全体を走査
- LoRAパス文字列の比較（文字列比較はコストが高い）

**改善案**:
1. **ハッシュベースの変更検出**
   ```python
   # LoRA追加時にハッシュを計算して保持
   self._loras_hash = hash(tuple((str(p), s) for p, s in self.loras))
   ```
   
2. **フラグベースの明示的通知**
   - LoRA追加時に`self._loras_dirty = True`を設定
   - forward()では単純なフラグチェックのみ

**推定改善効果**: forward()オーバーヘッド 5-10% 削減

---

### 2.5 【中優先度】 `lora_qwen.py` - QKV/GLU融合処理

**場所**: `_fuse_qkv_lora()`（850-987行）、`_fuse_glu_lora()`（792-847行）

**問題点**:
- テンソル連結（`torch.cat`）が複数回発生
- ゼロ初期化した大きなテンソルを作成してから部分コピー

**現状のコード例**:
```python
B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
B_fused[:out_q, :r] = B_q
B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
B_fused[out_q + out_k:, 2 * r:] = B_v
```

**改善案**:
1. **ブロック対角行列の直接構築**
   - `torch.block_diag()`の利用検討
   
2. **事前アロケーションの最適化**
   - テンソルプール導入でアロケーション削減

**推定改善効果**: LoRA合成時間 10-15% 短縮

---

### 2.6 デバッグログについて

> [!NOTE]
> **開発必須項目のため改善対象外**  
> 詳細なデバッグログ出力（`compose_loras_v2()` 内のキーインスペクション等）は、開発・デバッグに必須の機能であるため、速度改善の対象外とする。

---

### 2.7 【低優先度】 `nodes/lora/qwenimage.py` - 動的インポート

**場所**: `load_lora()` メソッド（97-110行）

**問題点**:
```python
spec = importlib.util.spec_from_file_location(
    "wrappers.qwenimage",
    os.path.join(lora_loader_dir, "wrappers", "qwenimage.py")
)
wrappers_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrappers_module)
```
- LoRA適用のたびに**毎回モジュールを読み込み実行**している

**改善案**:
1. **モジュールレベルでの事前インポート**
   - ファイル先頭で一度だけインポート
   
2. **キャッシュ済みモジュールの再利用**
   ```python
   if "wrappers.qwenimage" not in sys.modules:
       # インポート実行
   else:
       wrappers_module = sys.modules["wrappers.qwenimage"]
   ```

**推定改善効果**: LoRAノード実行時間 2-5% 短縮

---

### 2.8 【低優先度】 `qwenimage.py` (wrapper) - テンソルIDキャッシュ

**場所**: `__init__()` メソッド（55-61行）

**現状**:
```python
self._img_ids_cache = {}
self._txt_ids_cache = {}
self._linspace_cache_h = {}
self._linspace_cache_w = {}
```

**確認事項**:
- キャッシュが実際に活用されているか確認が必要
- キャッシュヒット率の計測が望ましい

**改善案**:
- 現状のキャッシュ実装が適切であれば変更不要
- LRU制限の追加でメモリリーク防止

---

### 2.9 【低優先度】 `lora_qwen.py` - モジュールパス解決

**場所**: `_get_module_by_name()` 関数（382-404行）

**問題点**:
- ドット区切りパスを毎回`split(".")`してトラバース
- 同一パスへの繰り返しアクセスでも毎回トラバース

**改善案**:
1. **パス解決結果のキャッシュ**
   - `@functools.lru_cache`の導入
   - 注意: モデル構造変更時にキャッシュ無効化が必要

**推定改善効果**: compose処理時間 3-5% 短縮

---

## 3. 優先度別まとめ

### 即効性が高い改善（推奨順）

| 優先度 | 改善項目 | 推定効果 | 実装難易度 |
|--------|----------|----------|------------|
| **最重要** | **LoRA重複読み込み排除** | **40-50%** | **低** |
| 高 | キーマッピング最適化 | 20-40% | 中 |
| 高 | LoRA遅延読み込み | 30-50% | 中 |
| 中 | ハッシュベース変更検出 | 5-10% | 低 |
| 中 | QKV/GLU融合最適化 | 10-15% | 中 |
| 低 | 動的インポート排除 | 2-5% | 低 |
| 低 | モジュールパスキャッシュ | 3-5% | 低 |

---

## 4. 備考

### 4.1 計測の必要性

上記の改善効果はコード分析に基づく推定値である。実際の改善を行う前に、以下の計測を推奨する：

1. **プロファイリング実施**
   - `cProfile`または`py-spy`によるボトルネック特定
   
2. **ベンチマークテスト作成**
   - 改善前後の定量比較

### 4.2 互換性への配慮

- 外部APIの変更は最小限に抑える
- 既存のワークフローが動作することを確認

### 4.3 この提案書について

- 本提案書はコード分析のみに基づく
- 実装変更は行っていない
- 実装着手前にユーザー様のレビュー・承認を要する

---

**以上**
