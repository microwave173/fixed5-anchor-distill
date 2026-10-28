# Fixed5-Anchor：从教师轨迹到 Qwen3.5-9B 训练

给 AI 的用法：**“请阅读这个仓库的 README，完成 Task 1 和 Task 2，汇报验收结果。”**
每个 Task 都有明确终点；完成指定任务即可，不自动执行后续任务。所有命令在仓库根目录运行。

## 这是什么版本

`动画 prompt → 总计划 → F1 → F2 → F3 → F4 → F5`。
画 F2–F5 时，每次同时看**总计划＋完整首帧＋完整上一帧**。输出自定义 Path3D JSON（3D 控制点），不是标准 SVG；没有 Python 工具调用或 refine。
教师生成与压缩均使用 DeepSeek high thinking。压缩保留原文已有的“公式 → 数值代入 → 中间向量 → 最终关键坐标”，允许 LaTeX，不额外补写几何检查。

| 已提供 | 数量 / 位置 |
|---|---|
| 完整教师轨迹 | **10 条**，每条 1 个计划＋5 帧，含原始与压缩 reasoning：[data/pilot10](data/pilot10) |
| 训练候选 prompt | **5,000 条**：[data/tasks5000/tasks.json](data/tasks5000/tasks.json)，不是 5,000 条完整轨迹 |
| 结果预览 | 下载仓库后打开 [data/pilot10/index.html](data/pilot10/index.html)；[质量检查](data/pilot10/REVIEW_ZH.md) |
| 生成 / 压缩 | [pipeline](pipeline)，默认不调用 API |
| 训练 | [training](training)，BF16 LoRA，默认 rank 64、alpha 128、1 epoch |

模型权重不在仓库内。10 条数据用于跑通流程，**不代表足以训练出高质量模型**。

## Task 1：新 GPU 服务器安装与预检

**输入：**一台 Linux NVIDIA GPU 服务器、Python 3.11+（建议 3.12）和本仓库；推荐先用单张 80–96 GB 卡验证。已使用的环境为 Python 3.12、CUDA 12.8 版 PyTorch；更小显存未保证。

```bash
git clone https://github.com/microwave173/fixed5-anchor-distill.git
cd fixed5-anchor-distill
# 系统需已安装 NVIDIA 驱动、python3、venv、git；检查 nvidia-smi 能正常运行。
bash scripts/setup.sh
source .venv/bin/activate
export MODEL_PATH="$PWD/models/Qwen3.5-9B"
hf download Qwen/Qwen3.5-9B --local-dir "$MODEL_PATH"
python scripts/preflight.py
```

**验收：**CUDA 可用，输出 GPU 型号/显存；最后显示 `Tokenizer and thinking prefix OK`。模型下载只做一次，之后每个新终端仍需激活环境并设置 `MODEL_PATH`。

## Task 2：用现成数据跑一次训练冒烟测试

**前置：**Task 1 完成；不需要 API key，不造新数据。

```bash
python training/prepare.py --data data/pilot10 --validation-tasks 2
torchrun --standalone --nproc_per_node=1 training/train_ddp.py --smoke
```

**产物与验收：**
- `training/prepare_report.json`：10 条轨迹，48 个训练样本＋12 个验证样本，`truncated=0`。
- `training/smoke/state.json`：`status=complete`，完成 **1 个优化器步**；loss、梯度为有限数。
- `training/smoke/adapter_check.json`：`all_finite=true`。
- 验证集按整条 task 隔离；不是按帧随机拆分。小样本测试不用于报告泛化成绩。

## Task 3：正式训练 1 epoch

**前置：**Task 2 通过；准备好已完整生成并压缩的数据目录。使用自有数据时替换 `--data` 路径与验证任务数。

```bash
# 现成 10 条的最小示范；大数据训练请改成自己的 runs/... 路径。
python training/prepare.py --data data/pilot10 --validation-tasks 2
# 配置在 training/config.json；configs/lora1ep.json 是默认备份。
torchrun --standalone --nproc_per_node=1 training/train_ddp.py
```

**验收：**`training/training/state.json` 为 `complete`；最终 LoRA 在 `training/training/adapter/`。完整 checkpoint 含 adapter、optimizer 和 progress，位于 `training/training/checkpoints/`，epoch 1 checkpoint 保留。

多卡将 `--nproc_per_node=1` 改为卡数。默认有效 batch 为 `1 × 8 × 卡数`；改变卡数会改变有效 batch。loss 按目标 token 归一化；输入不参与 loss，压缩 reasoning 与原始答案均参与。

恢复训练：在**数据、配置、卡数均不变**时重跑同一命令，会读取最新完整 checkpoint。仅保证从保存点恢复优化器与进度；没有保存全部随机状态，不承诺逐位一致。换数据/新实验请使用独立工作目录，不覆盖旧训练目录。不要把半成品轨迹送入训练。

## Task 4：重新生成 10 条教师轨迹

**前置：**联网环境、`httpx / numpy / Pillow`；需要有效 DeepSeek key，**会产生 API 费用**。不需要 GPU。Task 1 的环境也可以直接使用。

```bash
# 通过终端私密输入，不把 key 写入仓库或命令历史。
read -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
python scripts/new_run.py --out runs/pilot10 --concurrency 10
RUN_DIR="$PWD/runs/pilot10" python pipeline/bulk.py
```

**验收：**`runs/pilot10/results/state.json` 中 `counts.complete=10`，每条 `result.json` 的 `compression_complete=true`；各调用目录有 `compressed.txt`。打开 `runs/pilot10/dashboard.html` 看进度与 GIF，支持本地文件打开。

断线后使用同一 `RUN_DIR` 命令恢复，已完成结果复用。不要修改该目录里的 prompt/config/tasks；运行签名会阻止混用不同版本。失败任务不会冒充完成，需查看各 `result.json` 的原因后处理。

## Task 5：扩展到 5,000 条

**前置：**Task 4 验收通过；明确要生成全量并承担相应 API 费用。本步骤不会自动训练。

```bash
python scripts/new_run.py --out runs/full5000 \
  --tasks data/tasks5000/tasks.json --concurrency 2400
RUN_DIR="$PWD/runs/full5000" python pipeline/bulk.py
```

2,400 是同时在途请求的上限，不是吞吐保证。轨迹内部各帧有前后依赖；生成完成后才压缩。CPU 渲染另限 4 个并发。看板每 10 秒更新，请求报错有有限重试；大量失败时先排查额度、网络与 API 服务。完整 5,000 条约需 **30,000 次生成＋30,000 次压缩**，重试另计。

完成后先抽查各类别动画和压缩文本，再按 Task 3 准备数据并训练。数据分布、已知限制见 [数据说明](docs/DATA.md)，复现实测见 [验收记录](docs/VALIDATION.md)。
