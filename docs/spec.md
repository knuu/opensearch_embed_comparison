# OpenSearch に付属する埋め込みモデルの日本語性能を調査する

## これは何？

- OpenSearch には OpenSearch-provided pretrained models と呼ばれる，いくつかのモデル群がある
    - https://docs.opensearch.org/latest/ml-commons-plugin/pretrained-models/
- これらのモデルの日本語性能を調査したい

## 利用するデータセット

- JMTEB の Retrieval カテゴリのデータセットである jaqket を利用する
    - JMTEB: https://huggingface.co/datasets/sbintuitions/JMTEB
- 評価には test data のみ利用する
- 評価指標は公式の指標のうち NDCG@10 を利用する
    - 参考: 公式の指標は NDCG@10，Accuracy@{1,3,5,10}，MRR@10

## 利用するモデル

- OpenSearch-provided pretrained models
- 対抗モデル: Ruri-v3
    - Ruri-v3: https://huggingface.co/cl-nagoya/ruri-v3-310m

## 要件や仕様など

- local で docker-compose を利用して OpenSearch (3.3) を立てる
- index 登録や設定変更などは，あとで再現が可能なように，python script を作成する
- index template を用意して，index はそれを利用して行う
    - フィールド: id（自前で用意するか，データセットに付属する場合はそれを利用），text，embedding（ingest pipeline で作成）
- 検索には ANN を利用せず，exact kNN で実施する
    - 今回見たいのは速度ではなく精度であるため
- 1 つのコマンドで，1 つのモデルの評価指標が作成できるようにする
    - モデルを指定してスクリプトを実行 → （必要なら）各種設定の登録 → モデルに応じた index 作成 → 検索実行 → 評価指標の計算・保存
- テキストをベクトル化する処理は，利用するモデルによって，どこでベクトル化するかを変える
    - OpenSearch-provided pretrained models
        - テキストをベクトル化する処理は，opensearch 内で実施する
        - index 作成時（doc 登録時）にはおいて，先に ingest pipeline を登録しておき，そこでベクトル化も実施する
        - search 時には，search pipeline を定義しておき，そこでベクトル化した上でベクトル検索を実施する
    - 対抗モデル
        - テキストをベクトル化する処理は，opensearch の外側で実施する
        - sentence-transformers を利用する想定
        - mac で実行する想定なので，gpu として mps を使えるなら使う
