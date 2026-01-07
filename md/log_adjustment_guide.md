# LoRAデバッグログ出力の調整方法

本ドキュメントでは、`ComfyUI-QwenImageLoraLoader` ノードにおける「非対応LoRA形式のログ抑制機能」の手動制御方法について解説します。

## 概要

現在、フリーズ回避のために、**非対応形式（LoKR, LoHa, SD1.5用など）のLoRA**が検出された場合、自動的に詳細なキー検査ログの出力をスキップするようになっています。

これを強制的に解除し、全てのログを出力させたい場合は、以下の手順でコードを修正してください。

## 手順

1.  **対象ファイルを開く**
    *   ファイルパス: `custom_nodes\ComfyUI-QwenImageLoraLoader\nunchaku_code\lora_qwen.py`

2.  **該当箇所を検索する**
    *   以下のキーワードで検索してください（おおよそ 1207行目付近です）。
    *   検索語: `[USER REQUEST]`

3.  **コードを書き換える**

    **【現在の状態】（ログ抑制：ON）**
    ```python
    # [USER REQUEST] To restore full logs for unsupported formats, change the condition below to "if True:".
    _first_detection = _detect_lora_format(first_lora_state_dict)
    if _first_detection["has_standard"]:  # <--- ここが条件分岐になっています
    ```

    **【変更後】（ログ抑制：OFF / 全て出力）**
    ```python
    # [USER REQUEST] To restore full logs for unsupported formats, change the condition below to "if True:".
    _first_detection = _detect_lora_format(first_lora_state_dict)
    if True:  # <--- ここを "True" に書き換えます
    ```

    ※ `_first_detection["has_standard"]` を `True` に書き換えることで、形式に関わらず常にログ出力ブロックに入り、強制的に全てのキー検査ログが表示されるようになります。

## 注意点

*   **フリーズのリスク**: 非対応形式（数千個のキーを持つLoRA）に対してログを全出力すると、**ComfyUIのコンソールが数秒〜数分間フリーズする**可能性があります。
*   元に戻したい場合は、再度 `if _first_detection["has_standard"]:` に書き戻してください。
