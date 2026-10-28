# 可复现导出验收（2026-10-28）

已将本仓库的代码和 pilot10 数据独立复制到服务器目录，在已有依赖环境执行数据准备与单步训练；没有复用历史实验的 train.jsonl。

| 项目 | 实测 |
|---|---|
| GPU | 单张 RTX PRO 6000 Blackwell 96 GB |
| 基础模型 | Qwen3.5-9B，本地权重；fresh LoRA |
| 数据准备 | 10 tasks；48 train / 12 validation；按整条 task 分割 |
| 序列长度 | 最大 11,917；平均 6,264.3 tokens；0 截断 |
| 单步测试 | 8 个训练样本，1 optimizer step；另外验证 2 个样本 |
| LoRA | rank 64 / alpha 128；173,113,344 可训练参数 |
| loss / grad norm | 0.373393 / 1.014158，均为有限值 |
| GPU peak allocated | 27.50 GiB（不等于 nvidia-smi 总占用） |
| 时间 | 优化器步 33.3 秒；含加载/验证/保存总计 57.1 秒 |
| 最终状态 | complete；保存的 496 个 adapter tensors 全部有限 |

这验证了导出数据 → 分词/掩码 → forward/backward → optimizer → adapter 保存流程。
不是完整 epoch 质量评估，也不是在全新系统重装依赖后的测试；setup.sh 固定为本次已有环境的主要包版本。更长序列与不同卡型的显存/速度需重新预检。

本地看板已通过浏览器检查：file:// 方式能加载进度文件，显示卡片及任务列表。生产生成采用独立运行目录，不把运行中的数据自动提交至此仓库。
