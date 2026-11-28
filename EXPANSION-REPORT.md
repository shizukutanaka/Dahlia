# Dahlia 改善拡大レーン報告（Wave 1）

- 日付: 2026-11-28
- ブランチ: `exp/dahlia`
- ベース: `e26d69b` (`origin/master` / `origin/main`, v0.5.0)
- 仕様の正本: `docs/SPEC.md`（MVP スコープ・Non-Goals・フェーズ順は変更していない）

## 1. ベースライン（pristine な `e26d69b`）

実施した標準テスト:

| 対象 | コマンド | 結果 |
| --- | --- | --- |
| core | `(cd core && cargo test --no-fail-fast)` | 合格。27 passed / 0 failed |
| Python workers | `python3 -m unittest tests.test_workers tests.test_heic tests.test_evaluation` | 合格。33 tests / 10 skipped / 0 failed（127.102s） |
| Tauri | `(cd apps/studio/src-tauri && cargo test --no-fail-fast)` | 環境起因でビルド開始前に停止。`gdk-3.0` の `.pc` が見つからず、`gdk-sys` の build script が失敗 |

補足:

- baseline 実行時点の worktree には `.venv` と face/embedding モデルがなく、Python テストはシステム `python3` で実行した。`pillow-heif`・face/embedding モデル・mediapipe が無いため、従来どおり条件付きテスト 10 件が skip された。内訳は HEIC 系 6 件、実モデル評価 2 件、face model 1 件、mediapipe 1 件。後で元 clone `/home/hatch/workspace/dahlia` に環境済みの `.venv` とモデルが残っていることを確認し、最終検証では symlink で利用した（いずれも gitignore 済みの環境リソースで、ブランチには含めない）。
- Tauri の baseline 失敗はアサーション失敗ではなく、VM 差し替えで GTK/WebKit の開発用メタデータが無い環境問題だった。`apt-get update` が応答しなかったため、workspace 既存の復旧済み devlibs（`/home/hatch/tools/devlibs/root/usr/lib/x86_64-linux-gnu/pkgconfig`）を `PKG_CONFIG_PATH` に使った。
- 着手前に「ファイルの存在/不在・件数を固定するテスト」を grep した。Dahlia には Shoin/IZANAGI 型の repo 構造ガードは見つからず、件数固定は domain のテストデータ件数に限られた。

## 2. 反復で得た視点

マスク思考法:

1. 要件を疑う: Host/Origin 検証が問題になる常駐ローカルサーバーは Dahlia のアーキテクチャに存在しない。クラウド同期・サーバーは Non-Goals であり、追加しない。
2. 削除する: 新機能ではなく、既に保存している個人データを守る範囲に限定できる。
3. 簡素化する: 生成物ごとに別々の複雑な権限管理を足すより、アプリデータの境界（`app_data_dir`）を owner-only にし、 crown jewel である DB/backup を個別にも owner-only にする方が小さい。
4. 速くする: 性能上の新しい問題は今回の観測では最優先にならなかった。
5. 自動化は最後: 今回は自動化追加ではなく、既存テストへの回帰テスト追加に留める。

ソクラテス問答:

- なぜ敏感データだと言えるか: `embeddings.vector` は顔特徴量、DB には人物名・顔 bbox・写真パスが入る。さらに顔 crop・thumbnail・HEIC proxy は写真の複製である。
- 反証は何か: 原本の写真は app data にコピーされず、Windows の `%APPDATA%` は通常ユーザー専用 ACL である。それでも Unix/macOS では umask 次第で DB と複製物が group/other readable になり、portable 利用や権限の緩い HOME では実害がある。
- やめたら何が起きるか: 複数ユーザー環境で、Dahlia を使っただけで人物 Identity の DB と backup が他ユーザーに読める状態で残る。既存インストールでは、アプリ更新時に権限を直さない限り露出が継続する。

チェックリスト当てはめ:

- (a) 守りの欠落: **該当。秘密相当データと複製ファイルの権限が未指定だった。最優先に選んだ。**
- (b) 無言の失敗: `collect_images` は unreadable なディレクトリを空結果として黙って扱う候補が残っている。実在するが、今回は 1 件制約のため未着手。
- (c) 記録と実態のずれ: 今回の範囲で実測と矛盾する看板は確認しなかった。
- (d) データ消失リスク: DB backup・破損復旧・空クラスタリング防止は既に実装・テスト済みだった。ただし backup ファイル自体の権限は未保護で、(a) の一部として修正した。

## 3. 最優先項目と実測根拠

**選定項目: 顔特徴量 DB・その backup・写真/顔の複製物を含むアプリデータ領域が、Unix 系では所有者以外に読める権限で作られる**

実測:

- 実行環境の umask は `0007`。
- 修正前の `Db::open` で作られたテスト用 DB は `0640`、その親ディレクトリは `0770` だった（group readable/traversable）。
- 回帰テストを先に足すと、`database_and_backup_are_owner_only_on_unix` は修正前に失敗した（directory mode: left `0770` / expected `0700`）。
- よくある umask `0022` では、同じ仕組みでファイル `0644`・ディレクトリ `0755`（other readable）になる。

優先理由:

- 特徴量と顔画像は変更不可能な身体情報に近く、漏えると取り返しがつかない。
- SPEC の保存時暗号化のような判断待ち拡張ではなく、既存データ生成物の基本的な守りである。
- SPEC・MVP・Non-Goals・フェーズ順に抵触しない。新規依存も要らない。

## 4. 変更内容

- `dahlia_core::owner_only_file` / `owner_only_dir` を追加し、Unix では `0600` / `0700`、Windows では現行どおり ACL に任せる no-op にした。
- `Db::open` 後に DB 本体を `0600` に揃える。既存 DB が `0640` などでも起動時に修正する。
- backup について:
  - `backups/` を `0700` に揃える
  - backup ファイルを空のうちに `0600` にしてから `VACUUM INTO` で書き込み、終了後も再確認する
  - 既存 backup も起動時に `0600` へ修正する
  - backup 作成失敗時に空・中途半端な出力を残さない
  - 破損 DB の退避ファイルと復旧先 DB も `0600` に揃える
- Tauri 起動時に `app_data_dir` を `0700` に揃える。thumbnail、face crop、HEIC proxy、facemesh cache、depth map、log はこの配下なので、境界として owner-only になる。
- 回帰テスト `database_and_backup_are_owner_only_on_unix` を追加した。

新規の実行時依存追加: なし。

## 5. 変更後のテスト

| 対象 | コマンド | 結果 |
| --- | --- | --- |
| core | `(cd core && cargo test --no-fail-fast)` | 合格。28 passed / 0 failed（新規 1 件を含む） |
| Python workers | `.venv/bin/python -m unittest tests.test_workers tests.test_heic tests.test_evaluation` | 合格。33 tests / 0 skipped / 0 failed（54.802s、実モデル・HEIC・mediapipe を含む） |
| Tauri | `PKG_CONFIG_PATH=/home/hatch/tools/devlibs/root/usr/lib/x86_64-linux-gnu/pkgconfig CARGO_BUILD_JOBS=1 cargo test --no-fail-fast` | 合格。21 passed / 0 failed（unit 10、facemesh E2E 1、HEIC E2E 1、pipeline E2E 2、progress/cancel 6、space E2E 1。実モデル使用） |

`git diff --check`: 合格（whitespace error なし）。

## 6. 未検証・残課題

- Windows では `std::fs::set_permissions` では ACL を細かく操作できないため、今回の helper は no-op。Windows の実機では `%APPDATA%` 側 ACL を含めて実動確認が必要。
- macOS 実機では未実行。Unix 権限処理は共通だが、Tauri の `app_data_dir` 実パスで確認していない。
- Python worker の実モデル・HEIC・mediapipe テストは全件実行済み。Tauri 側の実モデル E2E 結果は最終テスト結果に記載する。
- unreadable な写真フォルダを import した時の無言スキップ（`collect_images`）は次点候補として残っている。
- TypeScript/React 側は未変更のため frontend build は今回の検証対象に入れていない。
