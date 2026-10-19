# vLLM KV 调度实验离线迁移包

本包包括 Python 3.11.16、vLLM 0.11.2、PyTorch 2.9.0+cu128、实验 venv 依赖、Stock/Optimized 源码 Overlay、Qwen3-1.7B 完整权重和 Tokenizer、CUDA Toolkit 12.8、Nsight Systems 2024.6.2、独立绘图 Python 环境、补丁、成本表及复现脚本。打包和安装都不下载依赖。

这是源机器的环境与代码快照。目标机器从全新 Run 开始；不携带原机器 raw 结果和完成状态，不会因原机器已运行而跳过目标实验。项目 reports 中的已有文档只是来源参考，不能当作目标服务器的实测结果。

## 目标服务器要求

- Linux x86_64、glibc >= 2.31。源机器为 Ubuntu 20.04 / glibc 2.31。
- 已安装可支持 CUDA 12.8 二进制的 NVIDIA 驱动。源机器驱动为 570.86.15；驱动属于宿主内核组件，本包不安装驱动。
- 本实验已在 RTX 3090 24GB 上验证。默认 Attention 路径使用 Ampere 或更新架构；其他 GPU 的可运行性以目标机器 GPU Smoke 为准。完整矩阵建议至少 24GB 显存，显存更小时需要调整预算。
- 系统有 GCC/cc、通常的 Linux C 开发头文件，用于 Triton 首次运行编译；以及 bash、tar、gzip、sha256sum。无需预装 Python、Conda、pip、CUDA Toolkit 或 Nsight。
- 建议准备至少 60GB 空闲磁盘，容纳外层包、内部压缩包和解压环境。安装后保留运行目录；如需节省空间，可移走外层 `.tar` 和已不再需要的 `packages/`，后者移除后无法再次执行包完整性校验。

若缺驱动或系统编译器，需要先通过服务器已有的离线系统软件源安装。不得把本机驱动目录或系统 glibc 覆盖到服务器。

## 传输和安装

将 `.tar` 与同名 `.tar.sha256` 一起复制到服务器。先进入两者所在目录：

```bash
sha256sum -c vllm-kv-offline-linux-x86_64-20261019.tar.sha256
mkdir -p /srv/experiments
tar -xf vllm-kv-offline-linux-x86_64-20261019.tar -C /srv/experiments
cd /srv/experiments/vllm-kv-offline-linux-x86_64-20261019
bash install.sh
source env.sh
```

`/srv/experiments` 可以替换为任意可写目录，不需要 root。先解压到最终安装位置，再运行安装。安装完成后如需换位置，应从原始包重新解压安装，避免保留旧 Conda 前缀。

安装脚本校验所有内部压缩包，解压运行环境，执行 conda-unpack，重新建立实验 venv，修复 Overlay 的二进制链接和项目绝对路径，并生成目标路径下的新源码冻结清单。原发布 SHA 与所有迁移修改保存在 `INSTALLATION.json`，不会将路径变化隐瞒成与原发布字节完全一致。

安装最后执行 CPU/导入/模型 SHA256 验证和绘图检查。GPU 生成测试需要显式执行，以免打扰服务器正在运行的 GPU 作业：

```bash
"$KV_EXP_PYTHON" "$KV_OFFLINE_ROOT/verify_offline.py" --gpu-smoke
```

该步骤启动单卡同步引擎与原生 CPU Offload，运行 2 个短请求；结果保存在 `data/offline_gpu_smoke.json`。CPU 检查记录在 `data/offline_verification.json`。

## 运行实验

每个新终端先执行本包根目录的 `source env.sh`，再进入项目：

```bash
cd "$KV_EXP_PROJECT"
"$KV_EXP_PYTHON" -m pytest -q tests
"$KV_EXP_PYTHON" benchmarks/run_quick.py --label offline_quick
```

完整实验：

```bash
"$KV_EXP_PYTHON" benchmarks/full_pipeline.py
```

后台运行可使用：

```bash
nohup "$KV_EXP_PYTHON" benchmarks/full_pipeline.py > "$KV_EXP_DATA/full_pipeline.log" 2>&1 < /dev/null &
```

查看状态：

```bash
"$KV_EXP_PYTHON" benchmarks/status_full.py
tail -n 30 "$KV_EXP_DATA/full_pipeline.log"
```

同一安装目录只启动一条完整流水线，避免竞争 GPU。流水线成功阶段和完整 Run 支持跳过恢复；失败时保留日志，先检查失败原因再用同一命令恢复。

完整流程包含正确性、HTTP、Nsight、130 个正式 Run、167 个开发扫描/消融 Run、独立长测与取消压力。结束后在 `project/reports/` 生成完整报告及 `v3_completion_receipt.json`；文件只有全部必做审计通过后才生成。Nsight 采集还受目标机器自身的容器/驱动性能分析权限约束。

## 不同 GPU 的成本重新校准

随包成本表来自 RTX 3090，本身不代表目标服务器的耗时。换 GPU、PCIe 拓扑或主要驱动环境时，建议在正式实验前重新校准。即使 GPU 相同，也可在空闲时重测：

```bash
bash "$KV_OFFLINE_ROOT/recalibrate.sh"
```

它顺序执行 Prefill、原生 H2D 测量和成本合并，更新 GPU 名称，重新冻结。若已有性能结果会拒绝执行，防止混用不同成本表；此时应使用另一份全新安装。现有驱动约定的成本文件名仍为 `qwen3-1.7b-3090.json`，内容按目标机器测量更新。

## 文件布局

```text
runtime/                  Python 3.11 与原生二进制依赖
analysis/                 Python 3.12 与绘图库
models/Qwen3-1.7B/        实体权重与 Tokenizer，无外部缓存链接
tools/cuda/               CUDA Toolkit 12.8
tools/nsight/             Nsight Systems
project/                  实验源码、补丁、测试与报告
data/environment/         重新建立的实验 venv
data/vendor/              Stock / Optimized Overlay
data/cost_models/         原始校准及有效成本表
data/raw/                 目标服务器的新实验数据
data/pipeline_v3/         阶段状态与日志
reference/                原 V3 实验方案
```

包不包含 Docker 镜像，不依赖 Docker，也不修改服务器系统安装路径。迁移后的性能应由目标服务器重新测量；本机进行的异路径验证覆盖环境导入、权重校验、符号链接、工具启动和绘图，不冒充目标服务器 GPU 验收。
