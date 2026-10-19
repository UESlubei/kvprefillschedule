# vLLM 0.32.0 / CUDA 12.9 单卡 RTX 3090 对比实验方案 V2

日期：2026-10-19。V2 基于本机已经完成的安装和真实 GPU 验证，取代 V1 中尚待验证的环境假设。目标是检验旧项目的准入优化在新版原生机制之上是否仍有增量收益。

**执行更新：NG/NT 已移植，已完成同步正确性、观测开销对照、12 Run 同步 Pilot 和 8 Run 默认部署 Pilot。默认路径正确性、原生重复与语义缓存检查已完成，本轮 55 Run 机制和 40 Run 默认部署正式矩阵完成；长测、7B 和满显存自动 KV 预算仍未执行。** 当前数据见 [新版实测报告](reports/latest_v0320/latest_comparison_report.md)。旧版 V3 已完成，其结果分开归档。

## 1. 已验证的环境

| 项目 | 实际配置 |
| --- | --- |
| GPU | 单张 RTX 3090，24 GiB，SM 8.6 |
| 宿主机驱动 | 570.86.15，未升级 |
| 系统工具链 | Linux / glibc 2.31，系统 CUDA 12.8 保留 |
| Python | 独立环境，3.11.16 |
| vLLM | 官方 `0.32.0+cu129` wheel，无项目调度补丁 |
| PyTorch / CUDA Runtime | `2.13.0+cu129` / `12.9` |
| Triton / FlashInfer | 3.7.1 / 0.7.0.post1 |
| Tag Commit | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| 模型 | `/mnt/ondeviceinfer/models/Qwen3-1.7B`，FP16 |
| 模型 Revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 环境 / 数据 | `/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320` |

采用官方 CUDA 12.9 二进制构建安装，未进行全量源码重编译。官方 Release 同时提供 cu129 资产，避免直接安装默认 CUDA 13.0 构建。[官方 Release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)

系统 `nvcc`、`nvidia-smi` 的 CUDA 显示与 Python 包携带的 Runtime 是不同口径；本实验以 `torch.version.cuda == "12.9"` 和真实 GPU 执行为依据。兼容结论限定于本次模型及覆盖路径，不扩展为所有算子、模型与 CUDA 功能均可用。

验证证据见 [环境验收报告](reports/latest_v0320/环境构建与验收.md) 和 [机器可读验收记录](reports/latest_v0320/validation_receipt.json)。

### 1.1 已完成验证及其边界

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| Torch CUDA | 通过 | FP16 GEMM、Pinned H2D/D2H、SM 8.6 |
| vLLM 同步 Eager | 通过 | 两条自然问答、固定 Token 生成、4096 Token GPU 前缀复用 |
| 原生 CPU Offload | 通过 | GPU 淘汰后恢复 4096 Host Tokens，H2D 469,762,048 Bytes，即 448 MiB |
| 原生默认执行路径 | 通过 | Async=True、torch.compile Mode=3、FULL_AND_PIECEWISE CUDA Graph，正常完成生成 |

Offload 验证为同步 Eager；默认路径验证未开启 Offload。因此“默认 Async＋Offload”的组合还须在 C 场景部署测试前单独验证。两条自然问答在三个模型测试模式下输出相同，这只是小规模冒烟，不能替代迁移后的正确性检查。

原生安装的 `scheduler.py`、`config/scheduler.py` 与官方 Tag 对应源码 SHA256 一致。没有复制旧版 Ready-holder 或 CPU 部分 Chunk 兼容补丁。

## 2. 核心问题与优化范围

第一优先级是 **GPU Cache 感知的有限窗口准入**：普通等待队列中，Cold 请求先分配 KV 可能驱逐后续 Warm 请求的可复用前缀；在受限窗口内优先处理可复用前缀较多的请求，减少实际 Prefill 与等待。其有效条件是队列确实积压、GPU KV 紧张且存在可保护的共享前缀。

第二优先级是 **GPU/CPU 分层代价感知准入**：在相同原生 CPU Offload 下，将 GPU 命中、CPU 连续 Ready 前缀、H2D 搬运与剩余 Prefill 成本一起纳入排序。必须与 GPU-only 准入额外对比，才能判断 CPU 信息本身是否有价值。

新版原生已经有完整输入长度预留、独立 KV 持有者等待队列和异步执行。项目收益可能被吸收，实验允许得到无收益或退化结论。[Scheduler 配置](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/scheduler.py)、[Scheduler 实现](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py)

代价感知抢占作为可选后续研究，不列入本轮必做矩阵。旧版部分抢占场景出现退化，优先迁移已有明确证据的准入优化。GPU/CPU LRU、KV 存储、原生 H2D 与模型计算继续复用 vLLM。

## 3. 对照组与归因

| ID | 定义 | 作用 |
| --- | --- | --- |
| O0 / O1 | 旧版 0.11.2 公共兼容基线 / 项目策略 | 历史参照，采用已归档结果 |
| N0 | 官方 0.32.0 原生代码，无项目补丁 | 新版主基线 |
| Nobs | N0 加只读机制观测 | 单独衡量观测开销 |
| NG | 0.32.0＋GPU Admission，保留原生抢占 | 项目 GPU 准入增量 |
| NT | 0.32.0＋Tier Admission，保留原生抢占 | 分层准入增量 |

NG/N0 回答 GPU 准入是否有效；NT/N0 回答分层方案整体是否有效；**NT/NG 才回答加入 CPU 成本信息是否额外有益**。C 中三组开启相同原生 Offload，其余场景关闭。

N0 使用统一外部 HTTP/SSE 计时和原生指标。项目 Probe、排序与必要统计的开销全部计入候选端到端耗时。原生缺少的内部计数写 `N/A`，不能补成 0。详细机制量在 Nobs 与相同观测配置的候选之间另做诊断，不以此替代 N0 主对照。

跨版本比较包含 Torch、CUDA、Kernel 和 Runner 的整体变化，不能将 N0/O0 的差距解释成调度算法收益。若原生路径需要修复，另列 Nfix，并保留未修复 N0 的失败记录；候选对 Nfix 的收益单独命名。

## 4. 固定预算与两条运行路径

### 4.1 同步 Eager 机制路径

| 参数 | 正式计划 |
| --- | --- |
| 模型 / 精度 / 并行 | Qwen3-1.7B / FP16 / TP=PP=1 |
| max_model_len | 8192；每条 Trace 的 Prompt＋输出不越界 |
| max_num_batched_tokens | 1024 |
| max_num_seqs | 16；本次环境冒烟使用 4，正式值仍需 Pilot 验证 |
| Prefix Cache / Chunked Prefill | 开启 |
| Async / Graph | `async_scheduling=False`、`enforce_eager=True` |
| GPU Block / Budget | 16 Tokens；640 或 1024 Blocks |
| 完整输入预留 / watermark | 原生实测默认 `True` / `0.0`，组间一致 |
| 抢占 | 各组保留原生默认 |

Qwen FP16 Full Attention 的 KV 为 **114,688 Bytes/Token**。GPU 640 Blocks 对应 10,240 Tokens / 1.09375 GiB KV；1024 Blocks 对应 16,384 Tokens / 1.75 GiB KV。这是人为控制的压力预算，不代表 RTX 3090 的全部可用显存。

使用 `num_gpu_blocks_override` 时，应记录最终 Block 数、实际 Token 容量和分配 Bytes。冒烟日志里“Available KV memory 12–13 GiB”属于覆盖前的估计，不能据此声称本次用了该大小的 KV。不同布局存在额外对齐时，以实际分配为准。

C 的原生连接器参数已经实测可用：

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 3758096384,
    "block_size": 256,
    "eviction_policy": "lru"
  }
}
```

CPU 目标为 32,768 Tokens / 3.5 GiB / 128 个 256-Token Chunks。这里 `block_size` 为 CPU 传输粒度，不是 GPU 的 16-Token Block。记录有效容量与已完成 Ready 区间；不启用磁盘、远程或其他缓存层。[原生 Offload 配置](https://docs.vllm.ai/en/v0.32.0/features/kv_offloading_usage/)

### 4.2 原生默认部署路径

本次 Qwen 冒烟解析为 Async=True、compile Mode=3、CUDA Graph=`FULL_AND_PIECEWISE`，日志使用 Model Runner V2。这些是特定模型配置的实际结果，正式 max_num_seqs=16 时重新保存完整解析配置。

原生组保留默认执行优化；候选若支持同样执行路径，在相同设置下配对。若同步自定义 Scheduler 会关闭 Async，则同时报告“同步候选 vs 原生默认”的实际差距，不能仅凭同步对同步的结果声称生产部署更快。

默认部署分别使用相同 KV 容量和相同总 GPU 显存上限两个口径，Graph 内存变化造成的 KV 容量差异单独报告。默认部署的 CPU Offload 组合须先通过正确性冒烟。

## 5. 移植与正确性门槛

1. 在独立候选目录移植 NG，不修改官方 N0 包。保存候选源码、补丁、上游与依赖 Hash；旧版冻结目录不变。
2. 检查新多队列语义，只重排普通、首次进入且尚未执行的 WAITING 请求。KV 持有者、已执行请求、加载中、恢复请求与其他状态作为屏障；保留原生资源优先级。
3. GPU/CPU Probe 必须只读，不改变引用计数、LRU、KV Events 或原生命中计数。CPU Prefix 必须连续 Ready，Pending Store、空洞与失败传输不当命中。
4. NG 先跑窗口 8、阈值 0、aging=0.05、max_wait_ms=2000 的开发起点。排序保护共享前缀，保持同分稳定；超时优先用于公平性保护，不作为 TTFT SLA。
5. 移植 NT 后重新测量 Prefill/H2D。比较 GPU 剩余计算路径与 CPU 恢复路径，按实际可执行恢复范围估计节约时间；CPU/GPU 重叠前缀不能重复计费。不得只按“GPU 命中＋CPU 命中”相加排序。
6. 若扫描参数，只用开发 Trace 做 Window=4/8/16、Threshold=0/0.5/0.75 的小规模单因子检查。正式前冻结选择，不根据正式结果继续调参。
7. 默认关闭策略应与对应官方运行路径行为一致；覆盖资源共享、部分 CPU Chunk、取消时在途 DMA、加载失败、双向回调、完成轮询及 KV 保护。
8. 同步和默认异步路径分别验证。自然文本逐请求及压力 Batch 都保存输出；发现差异时补同历史 Teacher-forced 概率/KL 检查，不能以“不同 Batch”直接忽略。

仅迁移必要接入，旧版兼容补丁逐项审查；本次原生 Offload 冒烟通过，不构成复制旧补丁的理由。

## 6. 新环境代价校准与观测

Prefill 校准覆盖输入 512/1024/2048/4096、缓存长度 0/1024/2048 的合法组合与 Batch=1/4，先预热，重复采样，保存分布与插值误差。之后按 Pilot 实际上下文补点，不在整个未覆盖区间无条件外推。

H2D 在原生连接器实际布局、Pinned Memory 和 Stream 路径下测量，覆盖 256/512/1024/2048/4096 Tokens；分离提交、GPU 拷贝与请求等待，保存真实传输字节数。旧 Torch 2.9 / CUDA 12.8 成本表仅供参考，不能直接用于新版正式计分。

已确认存在的原生指标：

- `vllm:prompt_tokens_cached_by_source`：按 `device` / `host` 等来源统计。
- `vllm:kv_offload_total_bytes`：按 `CPU_to_GPU` / `GPU_to_CPU` 统计实际搬运。
- 请求、排队、TTFT、E2E 与输出 Token 指标按本版导出名称归档。

本次 Offload 冒烟的部分传输耗时计数为 0，而 Host Token 和 H2D 字节计数已增长，因此 **0 不能解释为恢复没有时间成本**。须补专门计时或 Profile。缓存占用 Gauge 也不能单独证明目标前缀 Ready。

实际 Prefill / 重复执行 Token、PrefixLoss 和准入等待若原生未直接导出，使用经开销验证的 Nobs 诊断；不把逻辑 Prompt Tokens 当成实际计算 Token。

## 7. 负载、缓存初态与复现

复用旧 V3 的固定 Token Trace 生成逻辑，保持前缀组、长度、输出预算与请求到达方式；旧客户端/Runner 私有 API 不直接搬入新版。导出的最终 Trace 保存 SHA256，N0/NG/NT 使用同一文件。

| 场景 | 构造及验证重点 |
| --- | --- |
| A | GPU Warm/Cold 共享前缀混排；640/1024 Blocks；必须观察到队列积压和基线前缀损失，不能只串行重复命中 |
| C | GPU Warm、CPU Ready/GPU Cold、完全 Cold 混合；1024 GPU Blocks，CPU 32768 Tokens；保存真实 Host 恢复与 H2D |
| E | 无前缀复用，检验 Probe/排序开销 |
| F | 低并发闭环，检验不需要重排时的开销 |

旧参考到达率 A/E=3.2 req/s、C=3.5 req/s；新引擎先用开发 Trace 检查是否仍形成所需压力。历史复现保留旧速率；新版近饱和扫描单独命名，所有候选共用相同到达 Trace，不能逐策略各选有利负载。F 从并发 1 起步。

每 Run 独立服务进程、固定 CPU 亲和性与 OMP 线程，GPU 串行。先做 Kernel/Shape 预热，再用固定序列构造初态；重启清除上一 Run 缓存，但预热本身会新建 KV，不能假设预热后两层为空。

CPU Warm/GPU Cold 的构造：Seed Prefill → 确认 CPU Store Ready → 非共享压力流淘汰 GPU → 保留 CPU Seed → 正式请求真实恢复。当前冒烟已验证该构造可产生 4096 Host Tokens / 448 MiB H2D。正式仍需每 Run 校验，不能用固定睡眠替代 Ready，也不能用会实际计算的探测请求改变待测初态。

## 8. 快速验证与正式矩阵

### 8.1 首轮：12 Run × 64 请求，共 768 请求

| 场景 | 组别 | 种子 | Run 数 |
| --- | --- | --- | ---: |
| A / GPU640 | N0、NG | 811、823 | 4 |
| C / GPU1024 | N0、NT | 811、823 | 4 |
| E | N0、NG | 811 | 2 |
| F | N0、NG | 811 | 2 |
| 合计 | | | 12 |

正确性、代价校准和 Nobs 扰动检查另计。短跑用于验证策略是否激活、方向和开销，P99 不作为确定结论。C 的短跑不能证明 NT 优于 NG，正式三组对照补齐这一问题。

### 8.2 正式：55 Run × 256 请求，共 14,080 请求

正式配对种子固定为 1103、1129、1151、1171、1193，与开发隔离。

| 场景 | 组别 | 配对种子数 | Run 数 |
| --- | --- | ---: | ---: |
| A / GPU640 | N0、NG | 5 | 10 |
| A / GPU1024 | N0、NG | 5 | 10 |
| C / GPU1024 | N0、NG、NT | 5 | 15 |
| E | N0、NG | 5 | 10 |
| F | N0、NG | 5 | 10 |
| 合计 | | | 55 |

每个种子内轮换组别顺序，记录实际提交滞后，等待全部请求完成或按统一超时规则结束。不能仅等待固定短时间再丢弃未完成的慢请求。

### 8.3 默认部署：先 8 个短 Run，再决定扩展

A/C/E/F 各取一个预先固定的开发种子，对 N0-default 和对应候选做 64 请求配对，共 8 Run。C 先验证 Async＋Offload；候选不支持 Async 时如实标注同步。通过后再做 5 对正式重复，共 40 Run × 256 请求；另外的相同总显存比较独立计数。

对预先选定的 A640 / C 关键场景，只有正确性及正式结果支持继续验证时，补每 Run ≥2000 请求、5 对独立种子。长测种子在运行前冻结；退化与失败照常保留，不能筛选最好的一对作为结果。

## 9. 指标、统计与验收

主指标为 Client P95 TTFT、E2E、输出吞吐和完成率，P99 为补充。HTTP/SSE 首个非空内容定义 TTFT；Inter-chunk Latency 不直接冒称逐 Token ITL。模型加载、首次编译与 Graph Capture 单独报告，不混入稳定服务耗时。

配对统计按 Run 计算相对变化，再按配对 Run Bootstrap 给探索性 95% 区间。仅 5 对重复不应包装为高精度总体估计；不能将同一次 Run 内数百请求当独立硬件重复。延迟下降与吞吐增长分开报告。

同时展示 Warm/CPU/Cold 类均值与尾部、准入等待、调度 CPU 开销、重排次数、GPU/Host 命中、H2D 字节、错误和超时。重排获益若来自 Cold 等待增加，必须写明。2000 ms 超时优先仅为排序保护，需实测最长等待和完成率。

预先目标：E/F 吞吐退化不超过 3%、P95 TTFT 增加不超过 3%，同时报告区间；关键场景有同向配对证据、可解释的实际计算/搬运变化，才可宣称特定场景有效。开放到达负载下吞吐持平不能用于声称容量提升；容量需另测满足预先固定 SLO 的最大完成吞吐。

完整输入预留、watermark、Runner、Graph 等配置不得在候选与基线之间暗改。单独关闭完整输入预留可以作为消融，不能叫“新版默认基线”。

## 10. 较大模型与可选研究

首轮用 Qwen3-1.7B 保持与旧版同模型归因。可选本地 Mistral-7B-Instruct-v0.2 FP16，先冻结权重和 Tokenizer Hash、验证 Attention 与有效显存，重新计算 KV Bytes/Token 及可完成的单请求预算，再在一个关键场景跑 5 对。

不同模型重新生成合法 Token IDs，不沿用 Qwen 的 114,688 Bytes/Token 和成本表。未运行 7B 前，结论只适用于 1.7B。代价感知抢占、其他 Offload 后端和 KV 量化均作为单独因素，不混入本轮准入比较。

## 11. 执行状态与交付

| 阶段 | 当前状态 | 下一步交付 |
| --- | --- | --- |
| 官方 cu129 环境、依赖锁、GPU 兼容 | 已完成，独立审计通过 | 本文及环境验收报告 |
| NG/NT 新版移植、重新校准 | 已完成 | 独立扩展、正确性与成本表 |
| 12 Run 快速矩阵 | 已完成 | A/C 初步改善；E 同步尾延迟超过开发目标，正式复核 |
| 55 Run 正式机制矩阵 | 已完成并通过数据审计，质量范围见报告 | 逐请求数据、配对汇总和机制证据 |
| 默认部署、长测、可选 7B | 默认 Pilot 及 40 Run 正式已完成；长测/7B 未执行 | 独立报告，不混入 55 Run |

所有新版数据保存到独立数据目录，最终输出 `latest_comparison_report.md`、环境/补丁/Trace Hash、每 Run Manifest、SSE 时间戳、指标快照、失败日志和配对图表。执行入口需在移植阶段新增；不要用旧版 `benchmarks/full_pipeline.py` 直接运行新版。

旧版完整报告中的 GPU640 正式 P95 TTFT -19.05%、独立长测 -25.48% 仅是 **0.11.2 公共兼容基线上的历史效果**，不能写成已经在 0.32.0 得到相同收益。
