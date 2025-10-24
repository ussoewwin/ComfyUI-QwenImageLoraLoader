# ComfyUI-QwenImageLoraLoader インストールガイド

## 概要

ComfyUI-QwenImageLoraLoaderは、Nunchaku Qwen Imageモデル用のLoRAローダーです。このガイドでは、簡単なインストール方法を説明します。

## 前提条件

### 1. 必要なソフトウェア
- **ComfyUI** (公式版)
- **ComfyUI-nunchaku** (公式版)
- **Python 3.10-3.12**

### 2. 事前インストール
以下の順序でインストールしてください：

1. **ComfyUI** をインストール
2. **ComfyUI-nunchaku** をインストール
   ```bash
   git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku.git ComfyUI/custom_nodes/ComfyUI-nunchaku
   ```
3. **ComfyUI-QwenImageLoraLoader** をインストール
   ```bash
   git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader
   ```

## 自動インストール（推奨）

### 1. バッチファイルを使用したインストール

#### インストール
1. `install_qwen_lora.bat` をダブルクリックして実行
2. 自動的にComfyUI-nunchakuの`__init__.py`が修正されます
3. ComfyUIを再起動

#### アンインストール
1. `uninstall_qwen_lora.bat` をダブルクリックして実行
2. 元の`__init__.py`が復元されます
3. ComfyUIを再起動

### 2. バッチファイルの特徴
- **自動検出**: ComfyUIの場所を自動で検出
- **バックアップ**: 元のファイルを自動バックアップ
- **エラーチェック**: 必要なファイルの存在確認
- **簡単操作**: ダブルクリックで実行

## 手動インストール

### 1. ComfyUI-nunchakuの`__init__.py`を編集

`ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py` の最後に以下のコードを追加：

```python
# ComfyUI-QwenImageLoraLoader Integration
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
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = NunchakuQwenImageLoraStack
    logger.info("Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader")
except ImportError:
    logger.exception("Nodes `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` import failed:")
```

### 2. ComfyUIを再起動

## 使用方法

### 1. 利用可能なノード
- **NunchakuQwenImageLoraLoader**: 単一LoRAローダー
- **NunchakuQwenImageLoraStack**: 複数LoRAスタッカー（動的UI）

### 2. 基本的な使用方法
1. ComfyUIでワークフローを作成
2. `Nunchaku Qwen Image DiT Loader` でモデルを読み込み
3. `NunchakuQwenImageLoraLoader` または `NunchakuQwenImageLoraStack` を追加
4. LoRAファイルと強度を設定
5. 実行

## トラブルシューティング

### 1. よくある問題

#### ノードが表示されない
- ComfyUIを再起動してください
- `install_qwen_lora.bat` を再実行してください

#### インポートエラーが発生する
- ComfyUI-nunchakuが正しくインストールされているか確認
- ComfyUI-QwenImageLoraLoaderが正しい場所にインストールされているか確認

#### バックアップファイルが見つからない
- 手動で`__init__.py`を編集した場合は、元の内容に戻してください

### 2. ログの確認
ComfyUIのコンソールで以下のメッセージを確認：
- `Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader`
- エラーメッセージが表示されている場合は、依存関係を確認

## アンインストール

### 1. 自動アンインストール（推奨）
`uninstall_qwen_lora.bat` を実行

### 2. 手動アンインストール
1. `ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py` から追加したコードを削除
2. ComfyUIを再起動

## サポート

問題が発生した場合は、以下の情報を含めて報告してください：
- エラーメッセージ
- ComfyUIのバージョン
- インストール方法（自動/手動）
- ログファイルの内容

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
