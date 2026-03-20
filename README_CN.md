# SG-STGFormer 论文代码说明

这份文件夹是根据你的文章内容整理出来的随文代码包，已经放成一个可以继续补充和上传 GitHub 的结构。

## 里面已经有什么

- `sg_stgformer/`：论文核心模型实现
- `scripts/train.py`：训练脚本
- `scripts/evaluate.py`：评估脚本
- `scripts/predict.py`：单样本推理脚本
- `scripts/create_demo_data.py`：生成演示数据
- `scripts/create_ttedu_example.py`：生成 TTEdu 示例数据集
- `configs/ttedu_default.json`：按论文参数整理好的默认配置
- `configs/ttedu_example.json`：直接指向 TTEdu 示例数据的训练配置
- `MANUSCRIPT_SNIPPET.md`：可直接放进论文或回复审稿人的文字

现在项目里除了 `data/demo/` 之外，我还补了一个：

- `data/TTEdu_example/`：按你论文描述的 720 样本规模、16 关键点、81 帧、7:1.5:1.5 划分生成的数据集

## 我按论文还原的核心内容

- 自适应混合图构建
- 空间图 Transformer
- 因果掩码时间 Transformer
- 双向跨注意力时空融合
- 回归 + 分类 + 监督对比学习联合优化

## 目前默认参数

- 16 个关键点
- 81 帧
- 空间层 4 层
- 时间层 3 层
- 多头注意力 8 头
- 特征维度 256
- batch size 32
- epoch 100
- AdamW 学习率 `1e-4`
- weight decay `1e-2`
- 损失权重 `1.0 / 0.5 / 0.3`
- 温度参数 `0.07`

## 怎么用

先安装环境：

```bash
cd /Users/liuzhixin/Desktop/SG-STGFormer_论文代码
./setup_env.sh
```

跑一个演示流程：

```bash
./run_demo.sh
```

如果你想直接跑 `TTEdu` 示例数据，执行：

```bash
./.venv/bin/python scripts/train.py --config configs/ttedu_example.json
```

正式训练时，把你真实的骨架数据替换到 `data/` 目录，然后执行：

```bash
./.venv/bin/python scripts/train.py --config configs/ttedu_default.json
```

## 真实投稿前你还需要做的事

- 把演示数据换成你真正的 `TTEdu` 匿名骨架数据
- 不要把 `TTEdu_example` 当作真实实验数据写进论文结果
- 确认 16 个关键点顺序和你导出的姿态数据一致
- 重新训练一次正式模型
- 把 `MANUSCRIPT_SNIPPET.md` 里的 GitHub 地址改成你最后的仓库地址

如果你愿意，我下一步还可以继续帮你做两件事：

- 把这个文件夹直接整理成可上传 GitHub 的版本
- 帮你把论文里的 `Data Availability` 和 `Code Availability` 段落改成最终可提交版本
