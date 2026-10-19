# 环境和运行说明

## 1. 交付范围

这是原实验的源码仓库，不是随源码附带二进制环境的安装包。`SOURCE_MANIFEST.json` 核对从原项目复制的代码和配置；原始项目未被修改。参考成本表仅来自 RTX 3090，换 GPU、PCIe 或执行环境后应重新校准。

原项目中以下值是绝对路径。为了保持本次交付和实测代码一致，本仓库保留这些值；在另一台机器上运行前，需要调整对应脚本中的路径，或使用原目录布局。

| 路径 | 用途 |
| --- | --- |
| `/mnt/ondeviceinfer/experiments/vllm_kv_scheduling` | 旧版数据、venv、Overlay、成本表 |
| `/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320` | 新版独立环境、缓存和实验数据 |
| `/mnt/ondeviceinfer/conda-envs/vllm-0.11.2` | 旧版原生二进制基础环境 |
| `/mnt/ondeviceinfer/models/Qwen3-1.7B` | 模型权重和 Tokenizer |
| `/home/sdn/wjh/Ondeviceinfer/source/vllm_kv_scheduling` | 少量脚本和冻结校验引用的原项目目录 |
| `/home/sdn/anaconda3/envs/ljc/bin/python` | 新版最终绘图使用的分析解释器 |
| `/home/sdn/.local/bin/uv` | 新版最终依赖审计工具 |
| `/usr/local/cuda/bin/nsys` | 旧版最终性能分析工具 |

仓库根目录可直接作为项目根目录；大部分脚本使用 `__file__` 定位项目。剩余固定路径可用下列命令逐项检查：

```bash
rg -n '/mnt/ondeviceinfer|/home/sdn|/usr/local/cuda' \
  kv_sched benchmarks latest_v0320 offline_migration env.sh
```

更换路径或修改策略后，应使用全新的数据目录和冻结清单。旧版最终验收要求模型清单、校准验证及 Overlay 清单；新版最终验收还引用旧版冻结清单。只跑新版单项时可用 `run_http.py`，不必直接调用包含全部历史验收的 `finalize_comparison.py`。

## 2. vLLM 0.32.0

实测依赖：Python 3.11.16、官方 `vllm==0.32.0+cu129` wheel、`torch==2.13.0+cu129`、CUDA Runtime 12.9、Triton 3.7.1。驱动为 570.86.15，GPU 为 RTX 3090。完整版本记录在 `requirements/vllm0320-cu129-lock.txt`。

模型使用 Qwen3-1.7B FP16，Revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；模型文件需另行准备到配置路径。原实验采用官方预编译 wheel，没有全量源码编译。

在已安装依赖并准备好模型后，从仓库根目录运行：

```bash
source latest_v0320/env.sh
mkdir -p "$KV032_DATA"/{artifacts,cache,tmp,logs,manifests,pipeline,cost_models,raw}
mkdir -p /mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3
mkdir -p reports/latest_v0320
export PYTHONPATH="$PWD"

# 真实管理器的三个回归测试，不分配模型 GPU 内存。
"$KV032_PYTHON" -m unittest latest_v0320.test_admission -v

# GPU 空闲时重新测量成本，随后跑两类执行路径的正确性和 Pilot。
"$KV032_PYTHON" -m latest_v0320.calibrate
"$KV032_PYTHON" -m latest_v0320.pipeline \
  --stages correctness,correctness_default,observation,pilot,deployment_pilot
```

`calibrate` 调用实际 GPU 和原生 Offload Handler；其本地 Callable 配置用于校准进程，正常 HTTP 对照不依赖该配置。只检查代码或测试时不要启动 GPU 校准。

如果只查看调用参数：

```bash
"$KV032_PYTHON" -m latest_v0320.run_http --help
```

正式矩阵之前需要检查 Pilot 中的错误、初态、输出分歧和分类等待。原实验对分歧先运行 `teacher_forced.py`，严格检查失败后保留原记录，再运行 `quality_followup.py`、`assess_quality.py` 做补充诊断。不得直接复制旧完成标记或手工写入通过字段来跳过这些条件。

正式执行入口：

```bash
"$KV032_PYTHON" -m latest_v0320.pipeline --stages formal,deployment_formal
```

每 Run 启动独立 HTTP 服务；单 GPU 通过文件锁串行执行。完成后可调用 `audit_comparison` 和 `summarize`，绘图使用具有 matplotlib 的分析环境。不要在同一数据目录同时启动多条流水线。

## 3. vLLM 0.11.2

使用单独的 Python 环境，实测 Torch 为 2.9.0+cu128，vLLM 基础 Commit 为 `275de34170654274616082721348b7edd9741d32`。依赖锁文件为 `requirements/vllm0112-cu128-lock.txt`。

`benchmarks/prepare.py` 的 `INSTALLED` 指向该环境的 vLLM 安装目录。脚本校对原生源码，创建 stock/optimized 私有 Python Overlay，链接原 wheel 二进制；使用相同兼容补丁处理两组。

```bash
source env.sh
"$KV_EXP_PYTHON" benchmarks/prepare.py --compat
"$KV_EXP_PYTHON" -m pytest -q tests
"$KV_EXP_PYTHON" benchmarks/profile_costs.py --part prefill
"$KV_EXP_PYTHON" benchmarks/profile_costs.py --part transfer
"$KV_EXP_PYTHON" benchmarks/profile_costs.py --part merge
"$KV_EXP_PYTHON" benchmarks/capture_model_manifest.py
"$KV_EXP_PYTHON" benchmarks/freeze_sources.py
"$KV_EXP_PYTHON" benchmarks/run_quick.py
```

`--offline` 仅用于已经具有可信环境清单的安装，不用于跳过首次来源检查。完整矩阵、长测、HTTP、取消压力和最终报告由 `benchmarks/full_pipeline.py` 串行运行；最终验收还需要 Nsight 等工具及实际采样数据。

## 4. 离线迁移

`offline_migration/` 保存构建包、安装、路径重定位、Overlay 二进制链接修复、模型校验和 GPU Smoke 的源代码。历史包包含环境、权重和工具，约 14.23 GB，本仓库没有上传该包或分卷。

`offline_migration/README_离线迁移.md` 描述原包的用法。仅克隆源码不能直接运行 `install.sh` 安装整个环境，因为安装脚本需要另行构建得到的包内容。原迁移包主要对应 vLLM 0.11.2，不包含新版 0.32.0 全部环境。
