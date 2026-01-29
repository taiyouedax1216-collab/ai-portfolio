# ai-portfolio

# Bank Loan Default Prediction Model
銀行ローンのデフォルト予測モデル（二値分類タスク）

## 概要
SINGNATEで開催された三菱UFJ Basic Campコンペティションにおける作成モデルです。
中小企業向けローンのデータを用い、`LoanStatus`（1=デフォルト）を予測するモデルを構築しました。
不均衡データかつロングテールな分布を持つ特徴量に対し、対数変換やターゲットエンコーディング、および金融ドメイン知識に基づいた特徴量生成を行うことで精度向上を図りました。

- **評価指標**: F1 Score
- **モデル**: LightGBM (Single Model)
- **CV戦略**: 5-fold Stratified CV

## 開発環境
- **OS**: Linux (Ubuntu 22.04.4 LTS) / Google Colab
- **Language**: Python 3.12.11
- **Key Libraries**:
    - numpy 2.0.2, pandas 2.2.2
    - lightgbm 4.6.0, scikit-learn 1.6.1
    - scipy 1.16.1, statsmodels 0.14.5
    - seaborn 0.13.2, matplotlib 3.10.0

## データ分析とアプローチ

### 1. EDA（探索的データ分析）と課題
* **データの分布**: 金額系（`GrossApproval`等）や人数（`JobsSupported`）は強い右裾（ロングテール）を持つ。
* **地域性**: `CongressionalDistrict`（選挙区）は範囲が広く、件数の偏りが大きい。
* **ドメイン特性**: 固定金利よりも変動金利の方がデフォルト率が高い。また、法人（CORPORATION）の方がデフォルトしやすい傾向がある。
* **課題**: ラベル不均衡（デフォルトが少数派）であり、単純な学習では精度が出にくい。

### 2. 特徴量エンジニアリング
精度向上に寄与した主な特徴量は以下の通りです。

#### 変数変換
* **対数変換 (Log1p)**: 右裾の重い `JobsSupported`, `SBAGuaranteedApproval`, `GrossApproval` 等の分布を正規化。
* **Target Encoding (OOF)**: `CongressionalDistrict` に対し、情報のリークを防ぐため Out-of-Fold でターゲットエンコーディングを実施。
* **ラベルエンコーディング**: LightGBMと相性の良いLabel Encodingを採用。順序がある `BusinessAge` は順序を保って数値化。

#### ドメイン知識に基づく特徴量作成
* **比率特徴量**:
    * 保証率 (`GuaranteeRatio`) = 保証額 / 承認額
    * 従業員あたり融資額、借入負担比率など
* **相互作用**:
    * 金利負担 (`repayment_difficulty`) = 期間 × 金利
    * リスクコスト (`risk_andCost`) = 保証率 × 金利
* **集計特徴量**:
    * 地区（District）ごとの平均・中央値・標準偏差を集計し、元の値との差分や比率を算出。

### 3. モデリング詳細
* **アルゴリズム**: LightGBM (`LGBMClassifier`)
* **不均衡データの処理**: `class_weight='balanced'` を採用。
* **パラメータチューニング**:
    * 学習率: 0.03
    * n_estimators: 10000 (Early Stopping 400)
    * その他: `num_leaves=63`, `feature_fraction=0.8`, `lambda_l2=2.0`
* **後処理**:
    * OOF予測値と真値から Precision-Recall 曲線を計算し、F1スコアが最大となる閾値 (`best_thr`) を算出。この閾値を用いてテストデータを予測。

## 振り返りと今後の課題

### 成功要因
* EDAに基づき、金利・期間・保証率を組み合わせた相互作用特徴量を作成したこと。
* `CongressionalDistrict` の情報をOOFターゲットエンコーディングと集計特徴量で適切に取り込んだこと。
* 閾値最適化（Threshold Optimization）を行ったこと。

### 試行錯誤と課題
* CatBoostやXGBoostとのアンサンブル、Optunaによる探索も試みたが、LightGBM単体＋手動調整を上回らなかった。
* ローン種別（`Subprogram`）の表記揺れ対応が不十分だった可能性がある。

## 提出ファイルの作成手順
1. `train.csv`, `test.csv` を読み込み。
2. 前処理（不要列削除、エンコーディング、特徴量生成）。
3. StratifiedKFold(5) で学習し、OOFとTest予測確率を算出。
4. OOFで最適閾値を決定し、Test予測を二値化（0/1）。
5. 提出用CSVを出力。

```python
# 提出用コードの抜粋
submit = pd.read_csv("sample_submit.csv", header=None)
submit[1] = (predict >= best_thr).astype(int)
submit.to_csv("submission.csv", index=False, header=None)
