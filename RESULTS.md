# Week 2 Session — IMDB Top 500 情感分类实验结果

本仓库在 **`imdb_top_500.csv`**（500 条 IMDB 影评，`label`: 0=负面，1=正面）上对比了 **GloVe 词向量平均池化 + 逻辑回归** 与 **Hugging Face 上在 IMDB 微调过的 BERT / RoBERTa**。

---

## 1. 数据与运行环境（摘要）

| 项目 | 说明 |
|------|------|
| 数据文件 | `imdb_top_500.csv`，样本数 **500**，正负约各半 |
| GloVe | `tiny_glove.json`（50 维词向量） |
| Hugging Face | `HF_ENDPOINT=https://hf-mirror.com`（国内镜像加速下载） |
| 依赖安装（示例） | `pip install -r requirements.txt`（作业 Task 3）；`pip install -r requirements_hf.txt`（深度学习评测） |

---

## 2. Task 3：GloVe 平均向量 + 标准化 + 逻辑回归

- **划分**：`train_test_split(test_size=0.2, random_state=42)` → 训练 **400** 条 / 测试 **100** 条。  
- **脚本**：`task3_glove_lr.py`

| 指标 | 结果 |
|------|------|
| 训练集准确率 | **75.00%** |
| 测试集准确率 | **70.00%** |

**说明**：每条评论将分词后对应词向量取平均得到 50 维特征，再 `StandardScaler` 后训练 `LogisticRegression`。该方法**不建模词序**，且词表为 `tiny_glove` 子集，难度与深度模型不可直接横向对比「谁更强」，仅作传统基线。

---

## 3. Hugging Face：全量 500 条上的情感 pipeline 准确率

在 **不打乱顺序的同一 CSV 全表** 上对每条文本推理一次，与 `label` 逐条比较（**非** Task 3 的 train/test 划分）。

- **脚本**：`hf_imdb_eval.py`
- **模型**：
  - `textattack/bert-base-uncased-imdb`
  - `textattack/roberta-base-imdb`  
  二者均在 **IMDB** 上微调，与课程 CSV 来源一致，因此准确率显著高于 GloVe 基线。

### 3.1 总体指标

| 模型 | 前 50 条准确率 | 全部 500 条准确率 | 错误条数 |
|------|----------------|-------------------|----------|
| BERT (`bert-base-uncased-imdb`) | **96.00%** (48/50) | **98.40%** (492/500) | 8 |
| RoBERTa (`roberta-base-imdb`) | **100.00%** (50/50) | **98.40%** (492/500) | 8 |

**结论**：在 **全体 500 条** 上，两个模型 **总体准确率相同**；差别主要体现在 **前 50 条** 子集（数据顺序固定，子集较小，波动正常）。

### 3.2 结果文件（机器生成）

详细错例与片段已写入：

- `textattack_bert-base-uncased-imdb_imdb_results.txt`
- `textattack_roberta-base-imdb_imdb_results.txt`

### 3.3 BERT 错判样本（Review 编号 = CSV 从上到下的第若干条影评，`Review #k` 对应第 `k` 条）

| Review # | 真实标签 | 预测标签 | 备注（摘录含义） |
|----------|----------|----------|------------------|
| 10 | 0 | 1 | 西部片评论，含较多正面描写 scenery / adventure |
| 35 | 1 | 0 | 战后德国「火车」题材的文艺片评论 |
| 80 | 0 | 1 | 《星际迷航》前传剧集相关 |
| 126 | 1 | 0 | 德国电影 «Gespenster» 评论 |
| 159 | 0 | 1 | 「以为是纽约/喜剧」类反转叙述 |
| 357 | 1 | 0 | Petzold «Gespenster» 三部曲学术向评论 |
| 371 | 1 | 0 | 实验短片《OffOn》前卫风格 |
| 378 | 1 | 0 | 「spectacularly average / not bad」类温和褒贬 |

### 3.4 RoBERTa 错判样本（与 BERT 不完全重合）

| Review # | 真实标签 | 预测标签 |
|----------|----------|----------|
| 159 | 0 | 1 |
| 175 | 1 | 0 |
| 210 | 0 | 1 |
| 291 | 0 | 1 |
| 355 | 0 | 1 |
| 357 | 1 | 0 |
| 378 | 1 | 0 |
| 486 | 1 | 0 |

---

## 4. 方法对比小结（便于课程报告）

| 方法 | 评测设定 | 准确率（量级） |
|------|----------|----------------|
| GloVe 均值 + 逻辑回归 | 训练/测试划分，报告 **测试集** | **约 70%** |
| BERT / RoBERTa（IMDB 微调） | **全表 500 条** 逐条预测 | **98.40%** |

差异主要来自：**(1)** 评测协议不同（划分测试 vs 全数据）；**(2)** Transformer 在 IMDB 上端到端微调，表达能力与任务匹配度远高于「平均词向量 + 线性模型」。

---

## 5. 复现命令

```bash
# Task 3（建议使用清华 PyPI 镜像安装依赖）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python task3_glove_lr.py

# Hugging Face 评测（需安装 requirements_hf.txt）
pip install pandas -r requirements_hf.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python hf_imdb_eval.py
```

仅评测单个模型时可：

```bash
python hf_imdb_eval.py --models textattack/roberta-base-imdb
```

---

## 6. 免责声明

- 准确率依赖 **随机种子、划分方式与 CSV 顺序**；若课程更换数据或脚本，数值可能变化。  
- 本文件中的 GloVe 数字来自本地一次完整运行记录；HF 数字来自 **`hf_imdb_eval.py` 输出与同名 `*_imdb_results.txt`**，可与仓库内文件交叉核对。

---

*Generated for course submission — Week2 Session400*
