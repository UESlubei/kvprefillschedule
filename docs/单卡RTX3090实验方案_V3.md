# 基于 vLLM 的 KV Cache 分层感知调度与恢复代价优化

单张 RTX 3090 实验设计与分阶段实施文档 V3.0

编写日期：2026-10-19（Asia/Shanghai）。

**文档状态：本文件保留 V3 实验设计。0.11.2 实现及完整实验已经完成并通过最终审计，详见 [完整实验报告](vllm_kv_scheduling/reports/完整实验报告_V3.md) 和 [执行状态](vllm_kv_scheduling/PROGRESS.md)。快速探索保留为历史记录；后文模板未回填不代表实验尚未执行。**

依据：用户提供的《基于 vLLM 的 KV Cache 分层管理与代价感知调度优化》V2.0、引用对话中可读取的用户要求，以及本机实际环境。引用对话的助手正文只返回内容引用标记，无法读取完整文本，因此具体技术方案以附件为主，不将未读到的内容当作已确认结论。

## 1. 项目定位与研究问题

项目名称：面向单卡推理的 vLLM KV Cache 分层感知准入与恢复代价抢占优化。

英文名称：Tier-Aware Admission and Recovery-Cost-Aware Preemption for vLLM。

本项目围绕两个决策展开：

1. GPU KV 容量紧张时，Waiting Queue 中哪些请求应先准入，才能保住即将复用的缓存？
2. Running Request 分配 KV Block 失败时，选择哪个请求释放资源，实际恢复损失更低？

实现两个可以单独关闭的模块，并共享只读缓存探测和成本估计接口：

| 模块 | 决策对象 | 使用信号 | 输出 |
| --- | --- | --- | --- |
| GPU/Tier-Aware Admission | 有限窗口内可重排的 Waiting Requests | GPU 前缀、CPU Ready 区间、预计执行路径、等待时间 | 候选槽位的准入顺序 |
| Recompute/Restore-Aware Preemption | 可安全抢占的 Running Requests | 真正可释放的 Block、恢复代价、生成进度、当前缺口 | 单个 Victim |

本项目保留原生 KV 分配、引用计数、异步搬运、状态迁移和实际缓存命中确认流程。CPU 淘汰固定为现有 LRU，不新增第三项淘汰策略优化。

### 1.1 可检验假设

| 假设 | 必要条件 | 主要验证指标 | 不成立时应如何解释 |
| --- | --- | --- | --- |
| H1：GPU 感知准入能保留即将被破坏的共享前缀 | 同时存在排队、可复用 GPU 前缀及分配引发的淘汰 | 准入前缀损失、实际 GPU 命中、实际 Prefill Tokens、TTFT | 缺少这些条件时，排序可能只有开销 |
| H2：代价抢占能减少资源释放后的恢复损失 | 确实发生抢占，候选的共享程度、恢复代价或进度存在差异 | 实际 Recomputed Tokens、恢复耗时、ITL/E2E | 在同构请求中，各策略可能基本等价 |
| H3：CPU 分层感知优于只看 GPU | 存在已就绪、连续且实际可加载的 CPU KV，恢复代价具有差异 | CPU 实际命中、H2D Bytes、恢复耗时、TTFT | 3090 上搬运成本较高时，CPU 命中未必有利 |
| H4：联合策略可能存在交互效应 | 两个模块在同一 Trace 中都被触发 | 2×2 因子实验及调用计数 | Admission 减少抢占后，Preemption 的额外收益可能缩小 |

不预设必须提升某个百分比，也不把上游 PR 的测试数字作为本机预期。没有收益、出现退化或 CPU 路径不具备安全兼容性，均需要形成有证据的结论。

### 1.2 范围

首版支持单 GPU、单模型、Dense Full Attention、TP=PP=1、FCFS、同步 Scheduler、Prefix Caching 和 Chunked Prefill。

首版不扩展到 Priority、LoRA、多模态、Speculative Decoding、Hybrid/Mamba、分布式 KV 传输、PD 分离、主动 Session Prefetch、权重量化和新 Attention Kernel。Async Scheduling 作为后续独立兼容性阶段，不能在未经验证时默认启用。

CPU KV Offload 是前缀缓存保存与复用机制，不保证每个运行请求都有完整 CPU 副本，也不等价于任意请求的完整 Swap Out/Swap In。

## 2. 本机环境与模型选择

### 2.1 已确认的硬件和存储

以下数据来自本次只读检查，空闲量会随机器使用变化，正式实验仍需重新采集。

| 项目 | 本机情况 | 实验决策 |
| --- | --- | --- |
| GPU | 1 × NVIDIA GeForce RTX 3090，24576 MiB | 全部主实验在单卡完成 |
| GPU Driver | 570.86.15 | 固定记录，实验期间不升级 |
| 驱动报告的 CUDA 支持版本 | 12.8 | 与 Torch 构建版本分别记录，不等同于 Toolkit 版本 |
| CPU | Intel Xeon Gold 6152，22 核/44 线程 | 固定 Scheduler/Client CPU 亲和性，记录实际设置 |
| 主存 | 系统报告约 125 GiB，总可用约 116 GiB | CPU KV 从约 3.5 GiB 起步，保留充分余量 |
| NUMA | 本次 `lscpu` 显示 1 个 NUMA Node | 仍记录 GPU 拓扑和绑核结果 |
| PCIe | `nvidia-smi` 报告最大 Gen3 ×16；空闲时当前为 Gen1 ×16 | 用实测带宽建模，不按显卡宣传带宽估算 |
| 当前源码所在根分区 | 剩余约 21 GiB | 仅存文档、代码和小型汇总 |
| `/mnt/ondeviceinfer` | 剩余约 398 GiB | 环境、vLLM Checkout、原始数据和 Profile 放此处 |
| 性能工具 | 已发现 `/usr/local/cuda/bin/nsys` | 工具可用性及权限在实施时验证 |

PCIe 当前速率会随负载升降，空闲时的 Gen1 不能作为传输瓶颈结论。H2D/D2H 争用和 CPU Pinned Memory 行为都需要实测。

### 2.2 已有软件

优先基于以下环境复制出项目专用环境，避免改变已完成的 OnDeviceInfer 实验环境：

```text
/mnt/ondeviceinfer/conda-envs/vllm-0.11.2
Python:       3.11.16
vLLM:         0.11.2
PyTorch:      2.9.0+cu128
Transformers: 4.57.6
```

上述版本来自解释器和安装元数据检查。本次没有验证模型启动或 Offload 端到端兼容性。

本机还有 vLLM 0.12.0 环境，但主实验先使用 0.11.2，减少环境准备和 API 适配工作。若 Phase 0 发现无法解决的原生兼容性问题，再单独评估替换版本；替换后应重新建立全部可比基线，不能跨版本拼接结果。

### 2.3 主模型

**主模型采用本地 Qwen3-1.7B，直接用于功能和性能实验。** 小模型通过限制 KV Block 数同样可以稳定产生资源压力，不需要先换成 7B/14B。

```text
模型目录：/mnt/ondeviceinfer/models/Qwen3-1.7B
实际 Revision：70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
模型类型：qwen3
层数：28
KV Heads：8
Head Dimension：128
```

选择理由：本地已有权重，加载快；Full Attention 适合首版；FP16 权重为 KV、临时张量和诊断保留足够显存；主要研究变量是 Scheduler 决策，无需把模型规模作为前置门槛。

统一设置：权重 FP16、KV FP16、Greedy、固定输出长度、性能实验忽略 EOS。文本负载使用固定 Chat Template 并关闭 Thinking；精确控制长度的主实验使用预先生成的 Token IDs。两类结果分开呈现。

**可选规模验证：** 本地 `/mnt/ondeviceinfer/models/Mistral-7B-Instruct-v0.2`。本次配置检查显示 `sliding_window=null`，后续仍需核实实际 Attention 路径；模型目录缺少本次可确认的上游 Revision 时，先对配置、Tokenizer 和权重生成校验清单。该模型不进入首轮必做矩阵。

本地 Qwen3-14B FP16 不作为 24GB 单卡主实验；已有量化 Checkpoint 也不引入本项目的核心对照，以免增加量化 Kernel 和精度变量。

## 3. 固定版本与 API 修订

### 3.1 基础版本

本机 vLLM 的 `_version.py` 记录 `g275de3417`；上游 `v0.11.2` 标签经只读查询对应：

```text
275de34170654274616082721348b7edd9741d32
```

实施时以该 Commit 建立干净基线 Checkout，记录补丁、源文件哈希、环境锁文件和最终导入路径。不能只凭 `pip show` 的版本号认为当前 site-packages 一定未经修改。

建议保留三个可验证状态：

| 状态 | 内容 | 用途 |
| --- | --- | --- |
| Stock | 固定 Commit，保持原生逻辑 | 检查项目补丁默认关闭时的等价性 |
| Instrumented Baseline | 仅加入项目观测，策略全部关闭 | 性能对照的共同基础 |
| Optimized | 同一基线开启某个策略 | 测量策略差异 |

观测本身的开销用 Stock 与 Instrumented Baseline 单独比较。主要组间比较采用相同的观测设置。

### 3.2 相对 V2.0 的版本修正

以下为本机已安装的 v0.11.2 源码核查结果，不代表其他 vLLM 版本的接口。

| V2.0 中的概念或示例 | v0.11.2 实际情况 | 本项目处理 |
| --- | --- | --- |
| `cpu_bytes_to_use` | CPU Spec 读取 `num_cpu_blocks` | 根据模型 KV 大小换算 Block 数 |
| `v1/kv_offload/cpu/manager.py` | CPU Spec 在 `cpu.py`，LRU 在 `lru_manager.py` | 按本版本实际文件修改 |
| Offloading Scheduler 独立文件 | Scheduler/Worker 位于 `offloading_connector.py` | 通过该文件建立 Probe 接口 |
| `find_longest_cache_hit` 的示例返回值 | 本版本返回 Block Groups 和 Token 数两个结果 | 不复制新版本的三返回值伪代码 |
| `get_computed_blocks` 作为评分探针 | 会记录 Prefix Cache Stats | 候选评分绕过统计路径，调用核查后的只读底层查询 |
| Connector 真实命中查询作为评分探针 | `get_num_new_matched_tokens` 会先 `manager.touch` | 候选评分不能调用该方法 |
| 假设 `T_best=min(T_gpu,T_cpu)` 一定可执行 | 原 Connector 有命中时可能直接走 CPU Load | 评分使用真实执行路径，理想最小值单独作为诊断 |

CPU 配置字段由固定版本的 [CPUOffloadingSpec 源码](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/kv_offload/cpu.py) 支持；Ready 和 LRU 行为参见 [LRUOffloadingManager 源码](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/kv_offload/lru_manager.py)。

### 3.3 上游参考与项目边界

| 参考 | 本项目使用方式 |
| --- | --- |
| [#54625：Cache-Aware Admission](https://github.com/vllm-project/vllm/pull/54625) | 参考有限窗口和 GPU 前缀只读评分思想，在固定旧版本适配 |
| [#53723：Least-Computed-First](https://github.com/vllm-project/vllm/pull/53723) | 构建低复杂度 Victim 基线 |
| [#56897：Cost-Aware Preemption](https://github.com/vllm-project/vllm/pull/56897) | 构建按独占 Block、Token 数和进度评分的参考组 |
| [#57698：Recomputed Token Counter](https://github.com/vllm-project/vllm/pull/57698) | 参考准确区分实际重算与抢占时工作量的观测需求 |
| 原文引用的 #51787 | 作为 Offload 背景，本项目不移植其淘汰策略 |

这些 PR 不是 v0.11.2 的既有能力。实施前记录实际使用的 PR Head SHA；基于思想的适配应标为“参考策略实现”，只有固定并复现了对应补丁语义才能称“PR 复现”。不直接把多个面向新 main 的 PR 顺序 Cherry-pick 到旧版本。

## 4. KV 容量与实验起始配置

### 4.1 单卡 FP16 KV 预算

Qwen3-1.7B 每个 Token 的全层 KV Payload：

```text
KV_bytes_per_token
  = 2(K/V) × 28(layers) × 8(KV heads) × 128(head_dim) × 2(FP16 bytes)
  = 114688 bytes
  = 112 KiB
```

这是 TP=1、完整层 KV、无 KV 量化时的理论 Payload。元数据、对齐、Null Block、临时工作区和 CUDA Graph 不包含在内，正式结果以运行时分配为准。

| 对象 | Token 数 | 理论 KV Payload |
| --- | ---: | ---: |
| 1K Token 上下文 | 1024 | 112 MiB |
| 4K Token 上下文 | 4096 | 448 MiB |
| 8K Token 上下文 | 8192 | 896 MiB |
| 16K Token 上下文 | 16384 | 1.75 GiB |
| GPU Block，16 Tokens | 16 | 1.75 MiB |
| CPU Offload Block，256 Tokens | 256 | 28 MiB |

生成 Token 同样占用 KV；多个请求共用的物理前缀只计算一份物理存储。客户端并发数不能直接乘 Prompt 长度得到准确 GPU 占用。

### 4.2 GPU Budget

固定 GPU Block Size 为 16，通过 `num_gpu_blocks_override` 控制实验资源，而不是通过制造 CUDA OOM 构造压力。

| GPU Blocks | 理论 Token Slots | 理论 Payload | 初始用途 |
| ---: | ---: | ---: | --- |
| 4096 | 65536 | 7 GiB | 宽松容量对照 |
| 2048 | 32768 | 3.5 GiB | 中等容量候选 |
| 1024 | 16384 | 1.75 GiB | 主实验起点 |
| 640 | 10240 | 1.094 GiB | 高压力候选 |

表中容量尚未扣除保留 Block。初始化时必须确认实际可用容量大于引擎要求的单请求最大长度，并验证最长请求含输出能独立完成。

Low/Medium/High Pressure 由运行中的可分配空间、排队深度、分配失败和抢占频率定义，不能直接把上表某一行永久标为“高压力”。同一个 Budget 在不同共享比例下压力会不同。

本版本的 Free Queue 包含可被淘汰的缓存 Block。`KVCacheManager.usage` 主要反映当前不可分配的占用，不能等同于“有 Hash 的缓存占满程度”。缓存很多但无活跃引用时，usage 仍可能较低。

因此，GPU Admission 机制验证先将阈值设为 0；压力门控的正式实验再扫描阈值，并报告被跳过的 Step 比例。不能因为阈值阻止触发而误判准入策略无效。

### 4.3 CPU Offload Budget

主实验使用 CPU Block Size=256 Tokens，即 16 个 GPU Block。v0.11.2 配置中的 `block_size` 指这里的 CPU Offload Token 粒度。

| `num_cpu_blocks` | 理论 Payload | 用途 |
| ---: | ---: | --- |
| 32 | 0.875 GiB | CPU 容量受限对照 |
| 64 | 1.75 GiB | 小容量 |
| 128 | 3.5 GiB | 主实验起点 |
| 256 | 7 GiB | 容量扩展 |

如需要与 2/4/8 GiB 名义预算对齐，使用 `floor(budget_bytes / 29360128)`，并同时报告实际 Block 数和有效容量。不能把 128 Blocks 写成“4 GiB”。

CPU 容量按所有层合计。运行时应核实实际 Tensor Shape、是否 Pinned、分配后的 RSS 和系统可用内存；改变模型或 KV Dtype 后重新换算。

CPU Offload 粒度另做 64/128/256 Tokens 消融时，应固定总 Bytes，重新换算 Block 数，避免同时改变粒度和容量。

### 4.4 主实验起始参数

| 参数 | 起点 | 说明 |
| --- | --- | --- |
| 模型 / Dtype | Qwen3-1.7B / FP16 | 主实验全程固定 |
| KV Dtype | `auto`，运行时确认为 FP16 | 不引入 FP8/INT8 KV |
| 最大模型长度 | 8192 | 常规 Prompt 最大 6144，输出最大 256 |
| GPU / CPU Block Size | 16 / 256 Tokens | CPU Block 为 GPU Block 整数倍 |
| GPU / CPU Blocks | 1024 / 128 | 根据 Pilot 再固定压力档位 |
| `max_num_batched_tokens` | 1024 | 后续仅做 512/1024/2048 的单因素检查 |
| `max_num_seqs` | 16 | 客户端并发与此参数区分 |
| 客户端并发 | 1、4、8、16、32 | 32 可在引擎外或 Waiting Queue 形成排队 |
| Prefix Caching / Chunked Prefill | 开启 / 开启 | 除专项关闭测试外保持一致 |
| CUDA Graph | 首轮 `enforce_eager=True` | 先保证可解释性，Graph 另做对照 |
| Async Scheduling / Speculative | 关闭 / 关闭 | 首版正确性边界 |
| GPU Memory Utilization | 0.80 | 配合 Block Override；不靠它定义压力 |
| Admission Window | 功能验证 8，扫描 0/4/8/16/32 | 默认关闭时为 0 |
| Admission Threshold | 机制验证 0，正式起点 0.75 | 记录实际触发比例 |
| 随机种子 | 17/29/43/71/101 | 同一个种子用于组间配对 |

更长输入作为扩展实验：8K 输入时把最大模型长度提高到 12288，并重新验证 Budget；16K 输入时进一步提高最大长度与 Block 数。它们不进入首轮默认配置。

## 5. 系统结构与实现位置

### 5.1 决策边界

```text
Scheduler.schedule()
  ├─ Running Requests 申请 KV
  │    └─ 分配失败 → Victim Selector → 原有回滚/释放/重新排队 → 重试分配
  └─ Waiting Requests 准入
       └─ 一次有限窗口重排 → 原有 GPU Lookup / Connector Lookup / Allocate

Admission 与 Victim Selector
  └─ Read-only Cache Probe + Shared Cost Estimator

实际 H2D、D2H、Ready 确认与生命周期仍由原 Connector/Manager/Worker 处理。
```

Admission 不对 Running Queue 重新排序。Preemption Selector 不直接释放 Block，不直接清零请求进度。

### 5.2 建议修改范围

| v0.11.2 文件 | 计划修改 |
| --- | --- |
| `vllm/v1/core/sched/scheduler.py` | 两个策略接入点、公共非队尾回滚、计时和计数 |
| `vllm/v1/core/kv_cache_manager.py` | GPU 只读 Probe、只读资源缺口查询 |
| `vllm/v1/core/kv_cache_coordinator.py` | 必要的资源计数适配，优先复用现有接口 |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | 不改变生命周期的 CPU Probe、Ready/Load 完成观测 |
| `vllm/v1/kv_offload/lru_manager.py` | 只在无法复用纯查询时增加 Peek 接口 |
| `vllm/config/scheduler.py`、`vllm/engine/arg_utils.py` | 项目参数声明和映射 |
| 新增策略模块 | Admission、Preemption、Cost Estimator、项目指标 |

不在已安装 site-packages 中直接改文件开展正式对照；使用项目专用环境和可追踪的固定源码版本。导入路径、补丁 SHA 和干净基线校验属于 Phase 0 必做项。

### 5.3 建议目录

以下只是未来实施布局，本次不创建代码目录：

```text
/home/sdn/wjh/Ondeviceinfer/source/vllm_kv_scheduling/
  README.md
  configs/
  patches/
  benchmarks/
    generate_workload.py
    run_trace.py
    profile_prefill.py
    profile_transfer.py
    summarize.py
  tests/
  reports/

/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/
  environments/
  vendor/vllm/
  traces/
  cost_models/
  raw/
  profiles/
```

现有 `ondeviceinfer/` 推理框架用于背景参考，核心实验直接针对 vLLM V1，避免让两个独立 Runtime 的差异进入调度对比。

## 6. GPU/Tier-Aware Admission

### 6.1 重排资格与位置约束

只有满足以下条件的请求参与第一版重排：

1. 主调度策略是 FCFS，Prefix Caching 开启。
2. 状态为普通 WAITING，尚未计算、尚未被抢占。
3. 不在等待远端 KV、约束解码准备、异步加载等特殊状态。
4. 没有要求跳过 Prefix Cache Read，例如 Prompt Logprobs 路径。
5. 当前 Step 原本允许准入 Prefill。

取物理 Waiting Queue 前 K 个位置，而不是扫描到 K 个符合条件的请求。首版把不符合条件的位置视为屏障，仅在窗口内部各个连续的合格段内稳定排序，避免请求跨越特殊状态。窗口外保持原顺序。

每个 Scheduler Step 只做一次窗口决策，随后运行原生准入循环；不要每准入一个请求就重扫整个窗口。

### 6.2 GPU 只读查询

新增逻辑接口，例如 `probe_gpu_prefix(request)`：

- 复用固定版本 Coordinator 的连续前缀查询。
- 保留 Prefix Cache 开关、`skip_reading_prefix_cache` 和末 Token/Block 对齐约束。
- 不调用 Allocate、Touch、Free，不改变引用计数或 Free Queue 顺序。
- 不把候选查询写进真实命中统计，不产生缓存事件。
- 不改变 `num_computed_tokens`。

完整命中不能简单写为 `G=L`：本版本仍需计算用于生成 Logits 的末 Token，且 Block 对齐可能导致末 Block 重算。所有估计使用原生允许的有效缓存长度。

用查询前后快照验证无副作用：引用计数、Free Queue 顺序、Hash 索引、事件队列、Prefix Stats 和请求状态均应不变。

### 6.3 GPU-only 策略

合格段内按有效 GPU 连续前缀 Token 数降序排列，分数相同保持原到达顺序。正式准入时仍重新 Lookup 和 Allocate。

主要门控：窗口至少为 2、队列有多个候选、开启相关能力、usage 达到阈值。窗口关闭、低压力或不支持时，直接回到原有顺序。

门控用原生 usage 的语义，并记录 `gate_skip_reason`。先证明阈值为 0 时的机制，再判断阈值为 0.75 等设置是否适合本机负载。

### 6.4 Tier-Aware 策略

对每个候选得到 GPU 可用前缀 `G`、从 GPU 边界可连接的 CPU Ready 区间、实际传输 Bytes，以及成本模型预测。

定义：

```text
T_full = 无缓存时的 Prefill 成本
T_path = 按当前原生 Connector 规则预计执行的恢复 + Partial Prefill 成本
Saving = T_full - T_path
Score  = Saving + lambda × Waiting_ms
```

`Saving` 可以为负：CPU 搬运过慢时不能强行记为正收益。`lambda` 的单位为 ms/ms；初始候选值 0、0.05、0.1，在开发集选择，正式 Trace 不继续调参。

**恢复路径一致性要求：** 首版只修改顺序，不增加“拒绝 CPU 命中”的分支。如果原 Connector 将使用 CPU 命中，就按 CPU 路径估价。`min(T_gpu,T_cpu)` 仅记录为理想诊断值，不能假定 Scheduler 已具备选择两条路径的能力。

若以后增加按代价跳过 CPU 恢复，应作为独立可关闭扩展，增加新的 Baseline 和 Connector 状态测试，不与首版 Admission 收益混报。

### 6.5 公平性

Tier-Aware 的推荐组合为有限窗口、Aging 和超时保护。GPU-only 增加同样的超时保护作为公平配置对照；与纯参考排序的差异单独标注。

- 等待达到初始候选阈值 2000 ms 时，同一合格段内先按原到达顺序服务超时请求。
- 超时请求之间保持 FCFS，其余请求再按策略分数排序。
- 记录请求被越过次数、首次准入等待时间、各长度/冷热类别的 P95/P99 等待时间。
- 不跨越状态屏障，不突破资源约束，不覆盖用户显式 Priority。

超时保护只能约束调度优先级；资源暂时不可满足时，不能承诺 2 秒内一定准入。额外使用持续有 Warm Request 到达的有限压力测试，验证 Cold Request 不会被无限越过。

### 6.6 复杂度与 Fallback

每 Step 最多 K 次 GPU 查询及有限的 CPU 查询，排序为 `O(K log K)`；重建 Queue 的实际复杂度另行计时，不应把只扫描窗口等同于整个操作始终 `O(K)`。

成本模型缺失或不匹配时，Tier-Aware 降级到 GPU-only；CPU 能力不支持时也降级到 GPU-only；Prefix Caching 关闭或状态不合格时保持原生顺序。非法配置在启动时直接报错，不能悄悄转换。

## 7. Preemption 策略

### 7.1 对照策略

| 策略 | 选择逻辑 | 目的 |
| --- | --- | --- |
| Default | 保留固定版本原生选择 | 基准 |
| LCF | 最少已完成计算 Token，稳定 Tie-break | 低成本启发式对照 |
| PR-score | 独占 Block / 当前 Token 数 × 生成进度因子 | 参考 #56897 思路的适配对照 |
| Recompute-Aware | 满足资源缺口的候选优先，再按可释放 Block / 预计重算耗时评分 | 分离资源可行性和计算成本 |
| Restore-Aware | 使用已确认 CPU Ready 与保守 GPU 恢复信息估计真实恢复路径 | 验证两级缓存信息价值 |

PR-score 的 Token 定义应与实际采用的参考补丁核对并记录，不擅自替换成另一种 Token 数后仍称完全复现。

在无共享、相同 KV 布局的请求中，Block 数与已计算长度大致成正比，简单 `Blocks/Tokens` 可能近似常数。因此需要显式构造共享比例、生成进度或 CPU 可恢复程度不同的候选，不能预设复杂评分必然优于 LCF。

### 7.2 可释放资源

对请求持有的物理 Block 去重，统计“去掉该请求引用后，立即成为可分配资源”的 Block 数。

首版单组 Full Attention 下通常需要满足：

- 不是 Null/保留 Block。
- 当前唯一有效引用来自该请求；共享引用未清除时不算释放。
- 没有未完成搬运持有的额外保护或无法确认的 DMA 风险。

不能直接使用请求 Block 列表长度，也不能只检查 Hash 是否存在。跨多个候选累计释放时，共享 Block 的状态会变化，因此一次只选一个 Victim，原生释放后重新计算并尝试 Allocate。

### 7.3 当前资源缺口

从本次 Allocate 的同一组参数计算所需新增 Block，减去当前可分配 Block。复用现有 Coordinator 计数，保留 Lookahead、已分配尾 Block 等语义。

不要通过试分配或先释放候选来测量成本。不得将 `ceil(total_tokens/block_size)` 简化为所有情形下的新增需求。

选择顺序：

1. 建立安全可抢占候选集合。
2. 若存在单个请求足以满足当前缺口，优先在该子集评分。
3. 否则在有正释放量的候选中选择一个。
4. 执行原生回滚和释放，重新尝试 Allocate。
5. 仍失败则重复；没有安全进展时结束当前尝试，遵循原生可验证的等待路径。

### 7.4 恢复代价

定义 `T_r` 为请求实际已完成计算的范围；本轮尚未执行的预留 Token 不计入已完成成本。

Recompute-Aware 的保守版本假设被抢占后 GPU 前缀可能被覆盖，以无 GPU 恢复前缀估计 Prefill。Restore-Aware 在此基础上，只计入从序列起点连续、已 Ready 的 CPU 前缀，并使用 Connector 实际会执行的路径计算 Load 和剩余 Prefill。

```text
ProgressFactor = max(p_min, 1 - generated_tokens / max_output_tokens)
VictimScore    = ReclaimableBlocks / max(recovery_ms, epsilon_ms) × ProgressFactor
```

推荐初始 `p_min=0.05`，`epsilon_ms=0.01`；这只是数值稳定和进度启发式，不代表模型接近自然结束的概率。PR-score 对照保持其参考公式；项目公式的截断必须单独注明。

无校准模型时，Recompute-Aware 降级为 Token 近似版本并标为独立配置；不要将它混称为已校准时间模型。Restore-Aware 的 CPU 状态未知时采用保守成本，不能假定 Pending 已完成。

### 7.5 非队尾回滚

v0.11.2 的 Priority 分支已有对非队尾请求部分回滚的代码，可以作为审查起点；本项目需要把回滚整理为公共操作，不能只把 `running.pop()` 替换为任意下标删除。

必须覆盖：

- 已写入本 Step 的 `scheduled_running_reqs`。
- `num_scheduled_tokens` 和 Token Budget。
- `req_to_new_blocks`、Spec Decode 字段和 Encoder 预算等现有字段。
- Running Queue 遍历索引。
- Victim 就是当前分配失败请求时的退出行为。
- 请求状态、计数器、Waiting Queue 重新插入和事件。

用最小构造验证“本轮已选中一个前部请求，后续请求失败，Victim 又选回前部请求”的情况。所有成功输出中的 Token 预算和 Block 所有权必须一致。

### 7.6 CPU Offload 与抢占的兼容性门槛

**同步 Scheduler 不代表异步 D2H 已经完成。** v0.11.2 的请求结束延迟释放逻辑不能自动视为任意抢占路径也安全。

Phase 0/3 必须审查并测试：D2H Pending 时是否仍保护源 GPU Block，抢占和 Abort 是否通知 Connector，恢复后 Store 游标是否正确，以及已经清理的请求是否还能收到旧 Job 完成回调。

首版安全方案：将未完成 Store/Load、生命周期不明确的请求排除在项目 Victim 候选外。但这个过滤不能替代对原生 Default 路径的验证。

如果原生在该组合下也不能保证安全，暂停 CPU+Preemption 组合实验，保留已通过的 GPU-only 阶段和 CPU Admission 阶段，记录兼容性阻塞。可在独立版本分支适配上游修复后重建基线，不能回退到一个同样不安全的 Default 路径继续测性能。

只有兼容性通过后，Restore-Aware 和联合组才进入正式结果表。不得为完成矩阵而增加未经验证的 DMA 同步协议。

## 8. CPU Ready Probe 与状态一致性

### 8.1 Probe 应返回什么

概念接口需要包含：

```text
supported
gpu_prefix_tokens
cpu_ready_end_tokens        # 与 GPU 前缀衔接后可恢复到的绝对位置
cpu_pending_boundary        # 诊断信息，不参与可恢复评分
transfer_bytes              # 按真实分块映射计算
snapshot_step
unsupported_or_miss_reason
```

Ready 判断依据固定版本的完成状态：LRU 中条目存在且 `is_ready` 成立。本机已安装代码的 `lookup` 遇到 Miss 或 Not Ready 即停止；`touch` 是另一个会修改 LRU 的操作，评分探针不得调用它。

### 8.2 连续性和对齐

区分两个问题：

1. **准入：** GPU 已覆盖 `[0,G)`，CPU 只需从包含 G 边界的 Offload Chunk 开始，提供能连续衔接的 Ready 后缀。
2. **保守抢占恢复：** 如果不保证任何 GPU 前缀存活，则 CPU 必须从序列起点连续 Ready，不能使用孤立的后部 Chunk。

CPU Block 大于 GPU Block 时，G 可能落在 CPU Block 中间。Probe 必须镜像原 Connector 的 Chunk Hash 索引和跳过已在 GPU 的 Sub-block 规则。

例如 GPU 覆盖到 80 Tokens、CPU Block 为 256 Tokens，实际 Load 的 Sub-block 范围不能仅凭 CPU Chunk 数乘 256 推断；以固定版本 TransferSpec 的映射推导 Bytes，并与完成事件记录核对。

Ready 区间的上限也要满足原生末 Token 和最小有效加载长度规则。不能把全部命中 Chunk 无条件换成可免算 Tokens。

### 8.3 状态机约束

```text
Missing → Store Pending → Ready → Load Referenced → Ready → Evicted
                     └─ Store Failed → Missing
```

状态机是用于解释的简图，实际字段以原生实现为准。Probe 本身不推进任何状态，不创建 Load/Store，不增加引用，不刷新 LRU，不阻止淘汰。

评分与正式准入之间缓存可以失效。正式路径重新执行原生命中查询并锁定/加载；失效则按真实 Miss 继续 Prefill。不能缓存 Probe 结果后跳过真实确认。

Step 内探针结果可以用于排序，但不得在跨 Step 或释放/分配后作为资源有效性证明。记录预测到实际的间隔和失配原因。

### 8.4 必须区分的 CPU 指标

| 指标 | 含义 |
| --- | --- |
| Ready Predicted Tokens | 评分时看见的可恢复 Tokens |
| Actual Matched Tokens | 正式 Connector Lookup 接受的 Tokens |
| Loaded Tokens/Bytes | H2D 完成且用于有效恢复的 Tokens/Bytes |
| Prediction Miss | 预测命中后实际 Miss 或缩短 |
| Pending Excluded | 因 Not Ready 未纳入评分的边界/数量 |
| Load/Store Job Time | 提交至完成，不等同于纯 DMA 时间 |

Warm/Cold 标签需要按正式准入时状态复核。生成器中“打算构造 CPU Warm”不等于实际命中了 CPU。

## 9. Prefill 与 H2D 成本模型

### 9.1 模型形式

采用离线测量和轻量插值，不引入复杂学习模型：

```text
F(L,G,B) = 长度为 L、有效缓存到 G、负载档位 B 时的实际 Prefill 执行成本
H(bytes,blocks) = 固定提交成本 + 按字节及块数估计的 H2D 完成成本
```

对每个候选计算无缓存、GPU-only、CPU 恢复三条诊断成本，再按原生实际支持的路径得出 `T_path`。GPU 与 CPU 命中区间不重复计费。

成本模型只用于决策排序，不控制实际资源安全。不存在可靠样本或版本/模型不匹配时启用明确 Fallback。

### 9.2 Prefill 测量

首轮网格：

| 维度 | 初始采样 |
| --- | --- |
| Prompt 长度 L | 512、1024、2048、4096、6144 |
| 有效缓存比例 | 0、25%、50%、75%，按 GPU Block 对齐 |
| 请求 Batch | 1、4；8 仅在容量允许时扩展 |
| Chunked Prefill Budget | 固定 1024 |
| 预热 / 正式重复 | 每类 Shape 预热后，至少 10 次有效采样 |

采集 Model Runner/CUDA Event 的模型执行区间，并记录实际 `q_tokens`、`kv_tokens`、Batch 组成和执行次数。对于多个 Chunk，将本次 Prefill 的执行时间累加；不把排队时间、Tokenizer 或模型加载混入 F。

主模型先用单请求测得的表，再在 Batch=4 上检查排序误差。混合 Prefill/Decode 的运行时间不能直接解释为某一个请求的纯 Prefill 时间；需要单独标注共享批次并采用负载档位模型，或保守使用简单近似。

若整批同时进入且同等工作量，可用于验证批次总成本；不能无说明地将总时间除以 Batch Size 作为任意候选的边际成本。

### 9.3 H2D 测量分两层

第一层为 Pinned Memory 连续 Copy 微基准，使用 1、4、16、64、128、256 MiB，必要时增加 512 MiB。此结果用于确认 PCIe 基础带宽。

第二层使用原生 Offloading Handler 的多层 Block Copy，测量真实布局、Block 数、层数和 Job 提交开销。以这一层校准实际成本模型。

两层分别记录：

- CUDA Event DMA 时间。
- Host Submit 时间和提交到完成的 Wall Time。
- Payload Bytes 与 Bytes/s。
- 空闲 GPU 和同时存在 Compute/Store 时的差异。
- Pinned 状态、PCIe 活跃链路状态和测试线程设置。

原生实现是多层、多 Block 传输，不能仅用一个连续 `cudaMemcpy` 的带宽代替 Connector 端到端成本。微基准不算端到端优化证据。

### 9.4 训练、验证与误差

开发数据和最终 Benchmark Trace 分开。至少保留一组长度/缓存组合进行验证，不在测试集上持续调整表格或权重。

模型文件记录模型 Revision、vLLM Commit、GPU、Block Size、KV Dtype、Chunk Budget 和采集日期。不匹配时拒绝加载或明确降级。

记录 MAE、P50/P90 相对误差、候选排序准确率和 CPU/GPU 路径优劣判断准确率。相对误差分母使用 `max(actual_ms,0.1)`，同时给出绝对误差，避免近零成本放大比例。

可将 P50 相对误差≤20%、P90≤40%作为初始开发目标；达不到时先报告误差和决策影响，再简化模型或增加负载档位，不把该目标当作已实现精度。

## 10. Workload 与缓存初态构造

### 10.1 Trace 格式与 Token 控制

主实验采用固定 Token Trace，至少保存：

```text
request_id, scheduled_arrival_offset_ms, prompt_token_ids,
prompt_length, output_length, prefix_group, session_id,
seed, intended_cache_class, workload_version
```

Token IDs 是输入的事实依据。共享前缀在 Token 级拼接，避免“相同字符串经 BPE 边界编码后并不共享相同前缀”的问题。记录实际 Token LCP 长度，并对齐到 GPU/CPU Block。

不同前缀组在靠前的 Token 就分叉，避免只改变文末造成大段意外共享。随机 Token 合成只用于机制测试；文本应用负载使用固定文章、代码或问答材料，另行保存来源、Tokenizer 和模板版本。

### 10.2 负载集合

| 编号 | 负载 | 初始构造 | 主要用途 |
| --- | --- | --- | --- |
| W-A | GPU 共享前缀冲突 | 共享前缀 1K/2K/4K，独有后缀 128/512，Cold 先于 Warm 排队 | Admission 机制与前缀保护 |
| W-B | 长短请求混合 | 输入 512/2K/4K/6K，比例 40/30/20/10%；输出 64/128/256 | 抢占与恢复代价异质性 |
| W-C | GPU/CPU/Cold 混合 | 初始目标比例 30/30/40%，实际类别按准入事件复核 | Tier-Aware 对比 GPU-only |
| W-D | 多轮 Session | 8/16 Sessions、每个 4 轮、固定工具返回、错峰等待 | 缓存跨轮恢复 |
| W-E | 无共享前缀 | 相同长度分布，前部 Token 即分叉 | 退化行为和额外开销 |
| W-F | 低负载 | 并发 1/4、宽松 Budget | 低压力开销与公平性 |

W-B 输出长度首轮独立抽样；再用“长输入短输出/短输入长输出”对照检验进度因子的影响。所有 Prompt+Output 都必须小于 8192。

W-D 的主性能 Trace 固定历史 Token，不让不同策略生成的微小文本差异改变后续输入；真实使用模型输出继续下一轮的版本作为应用演示，不能承担严格配对的主结论。

### 10.3 GPU Warm 构造

1. 完成模型和 Shape 预热。
2. 提交指定 Seed 前缀，等待有效 Prefill 和请求结束。
3. 用只读 Probe 确认 Warm 前缀仍在 GPU。
4. 构造足够的 Running 占用或请求突发，让 Cold/Warm 同时进入 Waiting Window。
5. Cold 排在 Warm 前，Cold 分配确有淘汰 Warm Block 的可能。
6. 检查正式基线是否真的发生前缀缩短；没有发生则本次不能作为 H1 的机制证据。

仅在宽松环境先跑一条 Seed，然后逐条串行请求，通常不足以检验 Admission。

### 10.4 CPU Warm / GPU Cold 构造

1. 固定 Seed 请求完成 Prefill，观察对应 CPU Store 完成并 Ready。
2. 用与 Seed 不共享前缀的请求造成 GPU 淘汰。
3. 控制驱逐流量，避免同时超过 CPU 容量导致 CPU Seed 也被淘汰。
4. 验证目标请求 GPU 命中不足、CPU 连续 Ready 区间存在。
5. 待测请求进入相同排队场景，再观察实际 H2D 完成和使用。

不以固定睡眠或“请求已经返回”替代 Store Ready 确认。只清 GPU Prefix Cache 的操作也不能当作 CPU Cache 重置。

### 10.5 冷缓存与重复实验

首轮以重启引擎作为两级缓存统一重置方式，采用完全相同的 Shape Warmup、清理/驱逐和 Seed 步骤。任何用于预热的内容都不能意外成为正式请求的前缀。

如果后续实现缓存 Reset 工具，必须分别验证 GPU Hash/Free Queue、CPU LRU、Pending Jobs 和请求引用已回到预期状态，不能假设现有 GPU Reset API 会清除 Connector 缓存。

缓存初态按相同流程和实际可命中前缀校验，不要求物理 Block ID 相同。运行中不同策略自然产生不同缓存状态，这属于策略作用，不能为了强行保持一致而干预它。

## 11. 对照矩阵与实验规模

### 11.1 GPU-only 阶段

| 编号 | Admission | Preemption | CPU Offload |
| --- | --- | --- | --- |
| G0 | FCFS | Default | 关 |
| G1 | GPU-aware | Default | 关 |
| G2 | FCFS | LCF | 关 |
| G3 | FCFS | PR-score | 关 |
| G4 | FCFS | Recompute-Aware | 关 |
| G5 | GPU-aware | Recompute-Aware | 关 |

G0/G1 在 W-A 证明准入机制；G0/G2/G3/G4 在 W-B 比较抢占。GPU-only 完成后再加入 CPU，便于定位错误和收益来源。

### 11.2 两级缓存阶段

| 编号 | Admission | Preemption | CPU Offload |
| --- | --- | --- | --- |
| B1 | FCFS | Default | 开 |
| B2 | GPU-aware | Default | 开 |
| O1 | Tier-aware | Default | 开 |
| O2 | FCFS | Restore-Aware | 开 |
| O3 | Tier-aware | Restore-Aware | 开 |

比较关系：G0/B1 观察 Offload 自身影响；B1/B2/O1 分离 GPU-only 和 Tier-Aware Admission；B1/O2 观察恢复代价抢占；B1/O1/O2/O3 构成联合策略的 2×2 因子实验。

G0/B1 比较时不要同时改变 GPU Blocks、负载、Graph 或模型。CPU 引入的内存/传输变化属于 Offload 因素本身。

### 11.3 最小正式矩阵

避免把所有维度做笛卡尔积。建议在 Pilot 后固定两个有效压力档位：

| 实验 | 正式组合 | 配对重复数 | Run 数 |
| --- | --- | ---: | ---: |
| GPU Admission | W-A × G0/G1 × 两个压力 | 5 | 20 |
| GPU Preemption | W-B × G0/G2/G3/G4 × 两个压力 | 5 | 40 |
| Tier/联合优化 | W-C、W-D × B1/B2/O1/O2/O3 × 一个压力 | 5 | 50 |
| 低负载/无共享控制 | W-E、W-F × 对应 Baseline/联合策略 | 5 | 20 |

初始共 130 Runs，不含 Pilot、正确性和微基准。CPU 兼容性未通过时不执行相应组，并明确标为未完成。

每 Run 先采用 256 Requests 做运行成本估计和探索。重要尾延迟结论使用至少 2000 Requests/Run 的独立长测；短 Run 的 P99 只作为探索性指标。W-D 可通过增加 Session 周期达到样本量，但要保存同一长度和到达分布。

正式 Run 时长、排队 Drain 时间和数据规模以 Pilot 为依据，不预设所有实验能在固定数小时内完成。

### 11.4 参数扫描顺序

一次只改变一个维度，先小样本找范围，最终选定值再进行配对长测：

1. GPU Budget：4096/2048/1024/640。
2. 客户端并发：1/4/8/16/32，固定 Engine `max_num_seqs=16`。
3. Window：0/4/8/16/32，64 仅在开销可接受时扩展。
4. Threshold：0/0.5/0.75/0.9。
5. CPU Blocks：32/64/128/256，固定 CPU Block Tokens。
6. Aging：关闭/开启，并固定公平性保护的对照口径。

选参只用开发 Trace。保留一组前缀组、到达随机种子或长度分布作为最终测试集。

### 11.5 关键消融

| 消融 | 固定项 | 检验问题 |
| --- | --- | --- |
| 关闭 CPU Probe | Offload 仍开启、其他配置一致 | 分层元数据是否有价值 |
| Token 成本替代测量表 | 相同缓存探测和公平性 | 成本建模是否改善排序 |
| 关闭 Aging/超时保护 | 相同压力和冷热比例 | 效率收益是否由牺牲 Cold 等待换来 |
| Threshold=0 与门控 | 相同窗口 | 门控减少了多少开销和有效触发 |
| 忽略资源满足性 | 相同恢复评分 | 缺口优先规则是否减少无效多次抢占 |
| 关闭进度因子 | 其他评分相同 | 输出预算启发式的实际作用 |
| CPU Block 粒度变化 | 相同总 CPU Bytes | 传输粒度与 Ready 覆盖的影响 |

## 12. 指标定义与采集

### 12.1 Serving 指标

| 指标 | 统一口径 |
| --- | --- |
| Offered Load | 计划发送速率，单位 Requests/s |
| Request Throughput | 测量窗口成功完成请求数 / 墙钟时间 |
| Output Throughput | 测量窗口成功生成 Output Tokens / 同一墙钟时间 |
| Client TTFT | 实际提交请求到首个有效输出内容到达客户端 |
| Server TTFT | 请求到达引擎到首个输出 Token 可用；与客户端口径分开 |
| ITL | 相邻输出 Token 的生成时间差；优先由引擎事件获取 |
| TPOT | `(last_token_time-first_token_time)/(output_tokens-1)` |
| E2E | 提交到请求完成 |
| Initial Admission Wait | 到达到第一次有效准入，重准入另记 |
| Recovery Latency | 抢占到首次成功恢复执行，并拆分排队/Load/重算 |
| Completion/Failure | 成功、超时、取消、异常分别计数 |

SSE 可能一次返回多个 Tokens，网络 Chunk 时间差不能直接冒充逐 Token ITL。若只能得到 Chunk 时间，指标明确标为 Inter-chunk Latency；精确 ITL 使用服务端采集。

### 12.2 实际 Recomputed Tokens

不能在抢占时简单累加 `num_computed_tokens` 并称为“实际重算”。一部分 KV 可能重新命中 GPU/CPU；另一部分请求可能被取消而从未重算。

首版关闭 Async/Speculative，通过每个请求的**实际执行 Token 位置区间**统计：

1. 请求保存已经真实执行过的 Token 位置区间并集 `U`。
2. 本次模型实际完成执行的位置为 `[s,e)`。
3. `recomputed += length([s,e) ∩ U)`。
4. 合并 `[s,e)` 到 U。
5. GPU 命中和 CPU Load 只推进可用前缀，不作为一次模型计算写入计数。

例如先执行 `[0,512)`，恢复后仅执行 `[384,640)`，实际重算为 128 Tokens；再次执行 `[0,256)` 再增加 256，合计 384。不能第一次抢占就把 512 全算作实际重算。

执行区间以 Runner 成功执行结果为准，不能依据本 Step 尚未执行的 Scheduler 预留量。另记 `computed_tokens_at_preemption` 作为工作量诊断，两种指标不可混用。

```text
RecomputeRatio = 实际重复执行 Tokens / 实际执行 Tokens 总数
```

分母包含 Prefill、Decode 和重算，并同时给出 Prefill-only 重算统计。进阶 Async/失败输出场景还需区分物理执行浪费与最终有效提交，不能直接沿用同步版本记账。

### 12.3 缓存和前缀保护

至少记录：真实 GPU Hit Tokens、正式 CPU Matched/Loaded Tokens、GPU 分配与缓存淘汰、Ready/Pending/Miss、实际 Transfer Bytes。

Admission 的补充指标：

```text
PrefixLoss(r) = max(0, GPU_probe_before_window(r) - GPU_hit_at_actual_admission(r))
```

同一个请求第一次被探测到和正式准入的时间一起保存；重复探测不重复累加样本。该指标表示观察到的前缀损失，跨多个 Step 时还包含其他请求的淘汰作用。

在最小机制测试中结合 Block Hash 淘汰事件，验证损失是否由排在前面的 Cold 分配造成。正式长测中不把所有 PrefixLoss 都归因于单次 Admission。

缓存命中率同时报告分子和分母。GPU/CPU 指标采用明确、不重复的前缀区间，预测 Probe 不进入原生真实 Hit Counter。

### 12.4 调度器开销

用单调高精度时钟分别测量：

- 整个 `schedule()`。
- GPU Probe、CPU Probe、窗口评分和 Queue 更新。
- Victim 候选扫描、资源计数、成本估计。
- 策略调用次数、扫描请求数、扫描 Block 数、Fallback 原因。

报告 P50/P95/P99 Step Time 和各模块占比。详细请求日志只在诊断 Run 开启；正式 Run 使用聚合计数和有限采样，Baseline 与优化组设置一致。

额外记录 GPU 空闲 Gap、CPU 利用率、引擎/客户端 RSS、Pinned 内存、GPU 设备占用。vLLM 预分配 KV 后的设备占用不能解释为当前实际有效 KV 使用量。

## 13. 正式实验执行规范

### 13.1 Pilot

先运行 G0：以并发 1 验证最长请求能完成，再增加到 4/8/16/32。调整 GPU Budget，找到：

- 可复用但会被 Cold 分配破坏的 W-A 场景。
- 能稳定完成且存在足够抢占的 W-B 场景。
- 真实 CPU Ready、GPU 未命中并完成 H2D 的 W-C 场景。

建议机制长测至少观察到 30 次抢占或足够多的 PrefixLoss 事件；事件过少只说明场景未激活机制，不能据此比较策略优劣。极端反复抢占导致无法完成的配置只作边界诊断，不作为主性能结果。

### 13.2 预热与初态

每 Run 顺序固定：重启/验证源码 → 加载模型 → Shape Warmup → 确认两级缓存准备状态 → Seed/驱逐 → 初态只读校验 → 正式 Trace → Drain → 保存日志。

加载、编译、Graph Capture、预热和 Seed 不计入主测量窗口，但单独记录成本。冷/热实验采用不同清晰的初态定义，不能混合统计。

### 13.3 到达与排队

主要比较使用固定 Open-loop 到达 Trace，记录计划发送时间、实际发送时间和引擎到达时间，检查客户端是否跟不上。

发送速率从 Pilot 中确定，主压力档可围绕原生稳定吞吐的约 50%、80%、100%附近探索，最终以实测队列和抢占而不是比例命名压力。不要直接使用其他硬件 PR 中的 40 Requests/s。

Closed-loop 固定并发用于容量和交互式检查，结果与 Open-loop 分开。客户端并发上限如果导致延迟发送，必须报告 Client Queue，避免遗漏排队成本。

端到端计时从正式 Trace 首次提交至最后一个请求完成，包括 Drain；若报告稳态窗口，应同时给出窗口起止、已完成/未完成量和全量 Completion 结果。不得只挑某段高吞吐时间。

### 13.4 重复、顺序与统计

- 每组至少 5 次配对 Run，同一对使用相同 Trace Seed 和缓存准备流程。
- Baseline/Optimized 交替顺序；多组采用固定随机或 Latin-square 顺序。
- 每 Run 保存均值、P50/P95/P99，主分析对 Run 级指标做配对差值与 95% 置信区间。
- 不将同一 Run 中数千请求当成数千个独立硬件重复；需要请求级推断时使用按 Run/Session 分组的方法。
- 尾延迟按每 Run 分位数比较；合并样本分位数只能补充展示。
- 相近结果且区间跨零时，表述为未观察到稳定改善。

无测试异常不得事后删去最慢一轮。出现硬件干扰、引擎故障或客户端饱和时保留原始 Run，按预先规定的失效原因标记并补跑。

### 13.5 公平性与完成率

按 Prompt 长度、Cold/GPU Warm/CPU Warm、Session 分类报告等待时间和 E2E。平均 TTFT 下降但 Cold P99 大幅增加时，不能直接宣称整体改进。

初始控制目标：低负载/无共享下主要吞吐和延迟退化不超过约 3%，且没有超时或长期越过；小差异仍结合置信区间判断。该阈值是开发排查标准，正式报告公开实际结果，不为过线而选择性删除配置。

## 14. 正确性测试与故障注入计划

测试分为纯策略/Manager 测试、Scheduler 状态测试、GPU 端到端三层。优先覆盖会影响 KV 正确性和统计真实性的条件。

| 范围 | 必测场景 | 验收依据 |
| --- | --- | --- |
| 默认关闭 | Window=0、Default Victim | 固定输入下决策与原生一致，新增状态不影响结果 |
| Admission | GPU Warm 位于窗口内/外、同分稳定顺序 | 只改变允许的候选位置 |
| 状态屏障 | Blocked/部分执行/Preempted 请求混在窗口中 | 不跨屏障，不改变其状态和位置 |
| 能力关闭 | Prefix Cache 关闭、Priority、Skip Cache Read | 回到允许的原生行为或启动拒绝不支持组合 |
| 门控 | usage 低于/等于/高于阈值 | 边界正确，触发计数一致 |
| GPU Probe | 连续命中、末 Block、重复调用 | 引用、Free Queue、统计和事件全部无副作用 |
| CPU Probe | Missing/Ready/Pending/Failed、前缀中间缺口 | 遇缺口停止，Pending 不作为 Ready |
| CPU 对齐 | GPU 边界处于 CPU Chunk 中间 | Tokens/Bytes 与真实 TransferSpec 一致 |
| CPU 生命周期 | Probe 后淘汰、Abort、完成回调 | 真实 Lookup 安全 Miss，无陈旧资源复用 |
| LRU | 反复评分与不评分的同一访问序列 | 评分不改变 LRU 顺序和后续淘汰 |
| 可释放 Block | 独占、共享、重复引用、Null、传输保护 | 预测释放量与实际变化相符 |
| 资源缺口 | 单 Victim 足够、不足、多次抢占 | 每次重新尝试分配，无空转死循环 |
| 非队尾 | Victim 已在当前 Step 预留资源 | Token/Encoder/输出字段/索引完整回滚 |
| 当前请求 | Victim 就是本次分配失败请求 | 正确退出和重新排队 |
| 恢复模型 | CPU 比重算快/慢、模型缺失/不匹配 | 评分路径与实际 Connector 行为一致 |
| 实际重算 | 部分 GPU 命中、CPU 完整/部分恢复、多次重算 | Token 区间账本符合人工可计算结果 |
| 公平性 | 持续 Warm 到达、Cold 超时 | 超时优先规则有效，最终请求完成 |
| Offload 抢占 | Store/Load Pending 时发生分配失败 | 源 Block 不被提前复用，无错误回调 |
| 取消与异常 | 等待/运行/恢复时 Abort | 无引用泄漏、死锁和永久等待 |

### 14.1 端到端输出验证

固定小型中文、英文、代码和数学 Prompt 集；基线与优化组使用相同 Token IDs、模型、Greedy 参数及输出长度。

先在单请求下验证确定性 Token 一致；再在混合 Batch 和压力场景检查请求全部完成、输出长度、错误率和重复运行稳定性。若出现 Token 差异，进一步看每步 Logits、Top-1 Margin 和数值容差，区分 Batch 浮点差异与 KV 错读。

用 Teacher-forced 的同一 Token 序列核验缓存路径 Logits，可以避免后续文本分叉掩盖首次错误。不得用“调度可能改变数值”解释持续的大幅偏差。

长时间压力验证应包括反复抢占、跨轮恢复和取消；结束后检查 Running/Waiting、Pending Jobs 和引用持有者回到合理状态。

### 14.2 Async 扩展

只有同步版本验收后，才新增 Async Scheduling 对照，专门测试旧输出到达、占位符、Generation Epoch、Abort 和恢复计数。优先复用原生 Stale Output 协议；未完成时，在最终支持范围中明确排除 Async。

## 15. 分阶段实施与退出条件

### Phase 0：环境固定、基线和安全可行性

任务：准备专用 Checkout/环境；验证导入源码；记录 Manifest；用主模型跑通 Prefix Caching；确认 CPU Store→Ready→Load；建立 W-A/W-B/W-C；审查 Pending 搬运与抢占；建立统一最小观测。

交付：环境锁文件、源码/模型哈希清单、原生基线记录、调用链说明、GPU-only 与 CPU 组合的能力矩阵。

退出条件：最长请求可独立完成；能稳定激活缓存冲突和抢占；CPU 恢复确实有完成证据。若 CPU+Preemption 不兼容，注明阻塞，后续 GPU-only 开发可以继续。

### Phase 1：GPU Cache-Aware Admission

任务：GPU 只读 Probe、有限窗口分段重排、稳定 Tie-break、门控、关闭回退和超时保护；跑 G0/G1。

交付：Probe 无副作用测试、准入顺序测试、W-A 最小事件轨迹与独立性能结果。

退出条件：至少一个构造明确证明 Warm 前缀在 FCFS 下被破坏、在策略下得到保留；不能仅提交排序单元测试。

### Phase 2：GPU-only Cost-Aware Preemption

任务：LCF、PR-score、独占资源计数、缺口判断、公共非队尾回滚、Token 近似恢复模型与实际重算账本；跑 G0/G2/G3/G4。

交付：资源变化核对、非队尾回滚证据、实际重算统计、长短混合结果。

退出条件：多次抢占后请求全部完成、预算和引用正确；能够准确区分抢占时工作量与实际重复执行。性能无提升也应交付可解释结果。

### Phase 3：CPU Ready Probe

任务：CPU 元数据接口、Chunk 对齐、Pending/Miss、Probe 后失效、只读 LRU 验证；复核 Store/Load 与抢占边界。

交付：CPU 能力矩阵、无副作用测试、预测/正式匹配/完成加载三层事件记录。

退出条件：不提交任务、不增加引用就能安全估计可恢复范围；H2D Bytes 估计与实际传输映射一致。兼容性门槛不过时不进入 Restore-Aware。

### Phase 4：成本校准

任务：Prefill 和原生 Transfer Handler Profiling，建立带版本元数据的查表/插值模型，验证误差和排序质量。

交付：原始采样、模型文件、训练/验证划分、误差报告。

退出条件：成本能解释本机 CPU 恢复与重算的相对关系；不可靠区域有明确 Fallback；模型开销可测。

### Phase 5：分层与联合策略

任务：Tier-Aware Admission、Restore-Aware Victim、统一公平性、失配指标、两策略独立开关与同步端到端正确性。

交付：B1/B2/O1/O2/O3 功能验证及最小 2×2 结果。

退出条件：各组合明确支持/拒绝；CPU 失效时安全；不存在无法执行的理想路径估价；无错误 KV 复用和长期等待。

### Phase 6：正式 Benchmark、消融和 Profile

任务：固定开发后参数；运行配对正式矩阵与关键长测；执行控制组、必要消融和诊断 Profile；分析交互和局限。

交付：全部 Run Manifest、Trace、请求事件、聚合表、图和最终技术报告。

退出条件：每项结论可追溯至原始数据；性能、正确性、公平性和调度开销均有结果；未完成项明确列出。不能将 Phase 1/2 的成功写成全部两级缓存方案已完成。

## 16. Profile 与问题定位

### 16.1 CPU Scheduler

先用低开销聚合计时确认热点，再对短 Trace 使用 CPU Profile。重点看窗口查询、Queue 更新、Block 去重和多次失败分配。若 Python Probe 占据明显 Step 时间，先减小 K 或降低诊断频率，再讨论更复杂的缓存索引。

### 16.2 GPU 与 Transfer

Nsight Systems 仅采集短而有代表性的诊断区间，记录 Prefill/Decode、H2D/D2H、事件等待和 GPU 空闲 Gap。Profile Run 与主性能 Run 分开，保留相同 Trace ID 便于对应。

真实 Layer Copy 的小块数量、提交串行成本和双向 PCIe 争用，可能比纯带宽更能解释 CPU 恢复代价；GPU 空闲也可能来自 Scheduler CPU，而非模型 Kernel。

### 16.3 结果解释顺序

1. 策略是否触发，候选是否确有差异？
2. 是否改变实际 GPU/CPU 命中或 Victim？
3. 是否降低实际执行 Tokens、H2D 或恢复时间？
4. Scheduler 自身是否增加了相近或更大的开销？
5. Serving 延迟、吞吐、完成率与公平性是否同时支持结论？

不能从“命中率提高”直接跳到“整体吞吐提高”，也不能从“抢占次数减少”推导实际重算必然减少。

## 17. 结果表、图和验收标准

### 17.1 主结果表模板

所有数字待实测填写；每个表格固定同一模型、Budget、Trace、Arrival 和初态。

| 组别 | 实际 GPU Hit Tokens | CPU Loaded Tokens | Recomputed Tokens | P95 TTFT ms | P99 ITL ms | Output tok/s | Scheduler P95 ms | 成功率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Admission Only | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Preemption Only | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Combined | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

单独附上 Default/LCF/PR-score/Recompute/Restore 的抢占次数、每次释放 Blocks 和恢复成本；附上 Cold/GPU Warm/CPU Warm 的等待分位数。

### 17.2 交互效应

对同一指标 Y 使用绝对量计算：

```text
Interaction = Y_combined - Y_admission - Y_preemption + Y_baseline
```

延迟越低越好、吞吐越高越好，符号解释分别给出。报告各差值和配对区间，不直接相加百分比。

同时给出两模块的触发率和抢占次数：联合组中 Preemption 调用减少，本身可能是 Admission 的间接结果。

### 17.3 建议图

1. GPU Budget 对 TTFT、吞吐和实际重算的曲线。
2. Admission 前后有效前缀损失分布。
3. 每次 Victim 的释放 Blocks 与实际恢复耗时散点。
4. Prefill/H2D 预测与实测对比及误差分布。
5. Scheduler Step 时间组成。
6. Cold/Warm 等待时间 CDF。
7. 2×2 联合策略结果与置信区间。

只画能支撑结论的图，使用原始数据生成静态可导出的科学图，注明样本数、单位和配置。

### 17.4 最终验收

| 方面 | 完成要求 |
| --- | --- |
| 可复现 | 固定源码、模型、Trace、预算、种子和环境；一组命令可重建结果 |
| 默认行为 | 策略关闭时保持原生决策与正确性 |
| 缓存正确性 | Probe 无副作用，Pending 不被误读，实际路径重新确认 |
| 抢占正确性 | 非队尾回滚、共享 Block、取消和反复恢复均通过 |
| 可观测 | 实际重算、实际 H2D、调度 CPU 开销与预测失配可区分 |
| 机制有效 | 目标场景中有前缀保护证据；Victim 机制有可解释的代价差异 |
| 性能结论 | 有原生/参考策略/消融与重复测量，说明收益边界 |
| 公平性 | 分类等待、完成率和持续 Warm 压力测试均有结果 |
| 两级缓存 | Tier 与 GPU-only 明确比较；未通过兼容门槛则不得标完整验收 |

若结果显示收益不稳定，最终报告可以结论为“机制正确，但在该 3090/模型/负载下未观察到稳定性能提升”。不得把设计目标写成实测成果。

## 18. 复现资料与资源管理

### 18.1 每 Run Manifest

保存：run_id、组别、UTC 与 Asia/Shanghai 时间、源码 Commit/补丁 Hash、模型与 Tokenizer 校验、Python/Torch/CUDA/Driver、GPU 和 PCIe、Block 参数、内存预算、Scheduler 开关、Graph/Async/Spec 状态、采样设置、Trace Hash、到达率、并发、CPU 亲和性、初态检查和进程退出状态。

保存解析后的 Engine Config，不能只保存命令行，因为默认参数也可能改变行为。

### 18.2 原始数据

```text
raw/<run_id>/manifest.json
raw/<run_id>/requests.jsonl
raw/<run_id>/scheduler_steps.jsonl       # 聚合/采样，避免正式 Run 全量巨大日志
raw/<run_id>/cache_transfer_events.jsonl
raw/<run_id>/metrics.json
raw/<run_id>/server.log
raw/<run_id>/client.log
```

校准数据、Profile 和正式性能数据分目录。工作目录保留小型结果表和复现说明，模型、环境与 Nsight 大文件放 `/mnt/ondeviceinfer`。

每个阶段完成后统计数据量，必要时压缩已完成 Run；先保留聚合原始数据和 Trace，再决定是否保留全部大体积 Profile。不要复制现有模型权重建立项目。

## 19. 实施时使用的基线命令示例

**本节命令尚未执行，只是后续 Phase 0 的起点。正式运行前应使用项目专用环境，并确认 `--help`、启动日志、实际 KV Dtype 和源码导入位置。**

GPU-only 原生基线：

```bash
HF_HUB_OFFLINE=1 \
/mnt/ondeviceinfer/conda-envs/vllm-0.11.2/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /mnt/ondeviceinfer/models/Qwen3-1.7B \
  --served-model-name qwen3-kv \
  --host 127.0.0.1 --port 8010 \
  --dtype float16 \
  --kv-cache-dtype auto \
  --max-model-len 8192 \
  --block-size 16 \
  --num-gpu-blocks-override 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.80 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager
```

同一配置下开启 CPU KV Offload 时，额外传入以下 JSON：

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "num_cpu_blocks": 128,
    "block_size": 256,
    "eviction_policy": "lru"
  }
}
```

对应 CLI 形式：

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":128,"block_size":256,"eviction_policy":"lru"}}'
```

本版本 Async Scheduling 默认关闭；实施时必须在解析后的 Engine Config 中确认该值。以上不传入尚未实现的项目策略参数。

## 20. 计划新增的实验配置

以下是项目接口设计，不是当前 vLLM 已有参数。实现时再决定以项目配置文件还是原生 CLI 扩展暴露，并写入版本化 Schema。

```yaml
kv_scheduling:
  admission_policy: fcfs           # fcfs | gpu | tier
  admission_window: 0              # 默认关闭
  admission_threshold: 0.75
  aging_weight_ms_per_ms: 0.05
  max_wait_ms: 2000
  preemption_policy: default       # default | lcf | pr_score | recompute | restore
  cpu_probe_enabled: false
  cost_model_path: null
  unknown_cpu_state: conservative
  detailed_trace: false
```

所有策略必须可单独关闭。配置不支持的能力组合应明确报错或按文档定义降级，并在 Manifest 中记录最终生效策略。不能出现“请求 Tier-Aware，但整场都在 Fallback，结果仍标成 Tier-Aware”的情况。

## 21. 当前交付与下一次实施起点

本次交付仅为这份针对本机完善的实验文档。

已完成的准备性核查：硬件/存储、已有模型与 Revision、软件安装版本、固定 vLLM 标签 SHA，以及 GPU Cache/CPU LRU/Connector 关键接口的只读检查。

尚未进行：新项目代码、环境安装或复制、源码补丁、模型推理、CPU Offload 实测、单元测试、成本校准、性能 Benchmark 与 Profile。

后续按 Phase 0 开始，先获得可信的原生基线和 CPU 兼容性证据，再依次完成 GPU Admission、GPU Preemption、CPU Probe、成本模型和联合实验。
