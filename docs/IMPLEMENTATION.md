# 实现报告：KV 缓存查询、请求排序、兼容修复与实验问题

本文按代码实际行为说明实现。vLLM 0.11.2 和 0.32.0 分开记录；迁移包只作为运行环境交付过程说明。原始日志与模型没有上传，文中的日志路径是原实验的位置，不表示克隆仓库后已有这些文件。

## 1. 实现范围和代码入口

原生 vLLM 在请求执行时会查找可复用前缀。本项目提前查询等待请求的缓存，用查询结果调整等待顺序。

- 旧版独立算法：`kv_sched/policy.py`，包括配置校验、成本插值、稳定排序、抢占候选评分和已执行 Token 区间账本。
- 旧版运行时接入：`kv_sched/runtime.py`、`benchmarks/prepare.py` 与 `patches/scheduler-v0.11.2.patch`。
- 旧版共享兼容修复：`benchmarks/compat.py`、`patches/upstream-compat-v0.11.2.patch`。
- 新版：`latest_v0320/scheduler.py` 中的 `AdmissionMixin`，分别与原生 `Scheduler`、`AsyncScheduler` 组合。
- 新版实验：`run_http.py`、`calibrate.py`、`pipeline.py`、`audit_comparison.py`、`summarize.py`。

项目没有重写 KV 存储、Attention 算子、LRU 或 DMA。新版也没有移植旧版的抢占选择代码。

## 2. GPU 请求排序如何执行

`AdmissionMixin.schedule()` 先调用 `_kv_reorder()`，再执行 `super().schedule()`。

1. 读取等待队列前 8 个请求；没有至少两个候选时直接返回。
2. `eligible()` 只允许普通 WAITING 请求。已经计算过、被抢占过、跳过前缀读取、带异步占位或过期输出、可恢复的请求不参与排序。
3. `gpu_probe()` 调用 coordinator 的 `find_longest_cache_hit(block_hashes, num_tokens-1)`，读取最长可复用前缀。保留原生最后一个输入 Token 需要计算的语义。
4. GPU 策略直接以命中 Token 数为分数。`reorder_segments()` 只在连续可排序片段内操作，不能越过不合格请求；同分保持稳定。
5. 达到 2000 ms 等待阈值的请求，在所在片段内先按到达时间处理。该规则只改变优先顺序，不承诺 2 秒内得到首 Token。
6. 更新等待队列后交给原生代码分配资源、执行模型。

这里通过尽早使用已有 KV，间接影响后续缓存命中。代码没有根据未来访问频率给 KV Block 定价，也没有直接修改缓存淘汰策略。

## 3. CPU 请求排序如何执行

CPU 策略在 GPU Probe 后调用 `cpu_probe()`。目前只支持单组 FullAttention、原生 CPUOffloadingManager 和 LRUCachePolicy；其他结构明确报错。

代码按照原生 Offload Key 规则，从 GPU 命中终点继续查找 256 Token CPU Chunk。只把连续 Ready 的部分作为可恢复前缀；遇到缺失即停止。Pending 与正在加载的 Chunk 按第 6 节规则处理。通过 LRU 的纯字典读取查询，不修改访问顺序、引用计数或传输状态。

分数为：完整输入预计 Prefill 服务时间，减去利用缓存后剩余输入的预计 Prefill 服务时间，再减去 CPU→GPU 预计搬运时间，加上 `0.05 × 等待毫秒数`。该分数选择先执行哪个请求；实际恢复仍由原生 Connector 决定和执行，不声称实现逐请求最优重算/恢复路径选择。

成本表来自 `calibrate.py`：测量 14 个 Prefill 服务成本点和真实 Offload Worker 的 H2D Handler。排序用 Batch=1 点按缓存上下文近邻和剩余输入长度插值。服务时间包含调度、分发和采样，不是孤立 GPU Kernel 时间。默认异步编译路径沿用该冻结 Eager 参考表，没有单独验证预测误差。

## 4. 为什么要先解决旧版兼容问题

### 4.1 CPU/GPU Block 大小不同，部分恢复映射出错

**现象：** CPU Chunk=256 Tokens、GPU Block=16 Tokens，GPU 前缀结束在 CPU Chunk 中间时，真实恢复触发 `src_sub_block_count` 断言。

**原因：** 源和目标子块数量不同，跳过的源子块方向和映射数组长度处理不正确。

**处理：** 参考上游 #28951，按目标子块数量推导需要跳过的源子块，按实际目标长度分配映射。补丁同时应用于 stock 和 optimized。

**验证：** 用已知数值模式构造 CPU/GPU 块，检查跨两个 CPU Chunk 的中间区间恢复到指定 GPU Blocks，并验证反向拷贝，使用 `torch.equal` 比较实际数据。原日志为 `raw/cpu-smoke-v2.log`，测试在 `tests/test_native_managers.py`。

### 4.2 抢占时，异步 Store 尚未真正完成

**现象：** 不能仅根据已经提交 Store，就认定 CPU KV 已可读取；GPU 源块也可能在写回完成前被复用。

**原因：** 原版本部分 Pending 状态要到请求结束才确认，提交、完成和资源释放不在同一个时刻。

**处理：** 参考上游 #29870 的等待机制及相关 Store 提交顺序，在覆盖被抢占 Block 前提交排队 Store，并等待相应 CUDA Event；同步修改 Scheduler 的确认与清理逻辑。

**验证：** 使用实际搬运和 Pending/Ready 回归测试检查状态。此修复属于两组公共前提，不计为排序策略的性能收益。

### 4.3 非队尾抢占发布了尚未计算的 KV

**现象：** 某请求实际只执行到 31 Tokens，本轮预留第 32 Token 后被撤销；后续请求却能查到 32 Tokens 的命中。

**原因：** `allocate_slots()` 会提前登记本轮计划完成的完整 Block Hash。改变抢占对象后，选中的请求可能已经在本轮预留过资源；只归还 Token Budget 和释放 Block 会留下提前登记的 Hash。

**处理：** 在回滚分支调用 `rollback_cache()`，清除超出已确认计算进度的登记，再调用原生释放逻辑，保留已有效计算的前缀。

**验证：** `test_rolled_back_reservation_cannot_publish_uncomputed_kv` 用真实 KVCacheManager 验证回滚后只能命中已执行的 16 Tokens。对应修改在调度优化补丁中，不能只使用孤立 Victim 评分函数而省略回滚。

### 4.4 CPU 加载完成后丢失已生成历史

**现象：** 被抢占的请求从 CPU 恢复后，相同历史位置出现明显 Logit 偏差，曾达到最大约 12.19。

**原因：** 原路径加载后统一设为 WAITING，Worker 将其作为 NewRequestData 重新建立，已有输出历史被清空。

**处理：** 对已经发生过抢占的请求恢复 PREEMPTED 状态，走 ResumedRequestData 分支，保留生成历史。

**验证：** `test_preempted_cpu_load_is_resumed_with_generated_history` 检查恢复消息含原有输出 Token，且不被发布成新请求；再做固定历史数值检查。该错误与普通浮点差异分开处理。

### 4.5 取消请求时仍有 DMA 使用其 Block

**现象：** 请求已经逻辑结束，但加载目标或存储源还被异步任务访问；提前释放可能复用仍在搬运的内存。仅等待一个方向或重复 ACK 还会造成错误释放或引用不平衡。

**处理：** 结束请求延迟释放正在加载的目标，等待两个方向相关任务都确认完成；忽略重复 ACK。即使没有普通活跃请求，也继续轮询延迟完成的传输。处理已完成加载的持有者被内存不足队首阻挡的情况。

**验证：** `test_aborted_cpu_load_holds_gpu_destination_until_ack`、`test_cancelled_request_waits_for_both_jobs_and_ignores_duplicate_ack`、`test_completed_load_holders_can_advance_past_memory_blocked_head`。修复清单在 `configs/local_compatibility_fixes.json`，同样作用于旧基线和优化组。

## 5. 旧版策略和统计上的问题

### 5.1 共享 KV 导致“请求占了多少 Block”不等于“抢占能释放多少”

共享前缀由多个请求引用。旧版 Victim 评分读取引用关系，区分总持有和独占可回收 Blocks，并结合当前资源缺口、恢复成本和输出进度选择候选。回归测试覆盖共享但无法释放、资源不足和接近完成的候选。

该策略并未在所有场景加速：B/640 的 Recompute 使 P95 增加 14.34%、吞吐下降 1.49%；多轮 D 联合策略吞吐下降 3.76%。代码保留，结果按负向实验记录。

### 5.2 抢占进度不能直接当作实际重复计算量

`num_computed_tokens` 可能包含缓存恢复、计划进度或不同时间点的状态。使用 `IntervalLedger` 记录成功执行的 Token 位置区间，并计算新执行区间与既有区间的交集。例如执行 `[0,512)` 再执行 `[384,640)`，只增加 128 个重复 Tokens。

`tests/test_policy.py` 用独立位置集合与区间账本比较；运行时只在模型完成相应输出后提交区间。缓存恢复本身不记为模型计算。

## 6. 移植到 vLLM 0.32.0 时的问题

### 6.1 调度和 Offload 内部结构变化

旧版补丁不能直接用于新版本。新版已包含独立的持有 KV 等待队列、完整输入长度预留和 AsyncScheduler。

**处理：** 使用原生 `scheduler_cls` 扩展，两类调度器共用排序 Mixin。保留持有 KV 队列的原生优先级、完整输入预留=True、watermark=0、原生抢占与异步执行。新版 N0 直接使用官方包，不加载项目扩展。

**验证：** 每 Run 保存 `/server_info` 解析配置，检查实际 Async、预算及预留值；默认路径保留编译 Mode=3、FULL_AND_PIECEWISE CUDA Graph 和 ModelRunnerV2。

### 6.2 GPU 查询可能污染命中统计或事件

如果排序时调用会发布缓存事件的查询接口，就可能改变基线统计，甚至影响缓存状态。

**处理：** `gpu_probe()` 使用 coordinator 的只读最长前缀查询。

**验证：** `test_gpu_probe_preserves_native_resources_stats_events` 连续查询多次，检查 Block Hash、引用计数、空闲数量、空闲链表顺序、缓存统计和事件都不变，并检查最后一个缓存 Token 的重算边界。

### 6.3 CPU 部分 Ready 后遇 Pending，不能直接报部分命中

早期实现曾按连续 Ready 前缀推测可恢复，但原生在后续匹配 Chunk 为 Pending 时会延后整次查询。

**处理：** 一旦碰到匹配的 Pending，就返回无额外 CPU 命中；共享 Chunk 已处于加载中时也不把它算成立即可恢复。旧开发结果单独归档，正式数据来自修正后的冻结代码。

**验证：** `test_cpu_native_lru_probe_stops_at_pending_and_keeps_recency` 使用真实 CPU 管理器，覆盖 Ready/Pending/缺失/In-flight，并确认引用与 LRU ranks/heap 不变。

### 6.4 成本表不能直接复用旧版本测量

Torch、Runner、搬运结构和调度发生变化，旧表不能代表新环境。实际 Handler 接口对 Block 分组、索引和本地 Callable 序列化也有要求。

**处理：** 使用新环境、真实原生 Offload Worker 重新测 H2D；Prefill 单独测一输出服务耗时。修正分组和索引参数，本地 Callable 只在可信校准进程中启用。

**验证：** 例如 2048 Tokens / 224 MiB 的 H2D 约 21.94 ms，拟合固定开销约 0.566 ms、带宽约 11.0 GB/s。成本文件和版本、模型 Revision 一起冻结。此表是排序近似，不能推导纯 Kernel 加速。

## 7. HTTP 实验和数据导出的问题

### 7.1 关闭服务时调度诊断没有写出

原生 Core 在关闭流程中可能被强制结束，依赖 `shutdown()` 导出不可靠。

**处理：** 计时结束后调用原生双缓存 Reset，在该接口中一次性导出内存统计；关闭时只补写尚未生成的文件。防止后续取消测试和 Reset 覆盖测量区间的数据。

**验证：** 检查每个候选 Run 的 `scheduler.json`、完整区间确认和计数。写文件与 Reset 不计入性能窗口，查询、排序和内存记账开销仍计入。

### 7.2 突发请求恰逢 HTTP 空闲连接超时

一个 Pilot 原生基线出现 1/64 ServerDisconnected，发生在 5 秒边界附近；该轮开发 Run 整体归档。

**处理：** 后续 Pilot 和正式组统一原生 HTTP Keep-Alive 为 120 秒，基线和候选相同。没有剔除单个失败或慢请求来重新计算更好结果。

**验证：** 保存 `runtime-environment.json` 与错误数；正式产物要求请求数完整且错误为零。

### 7.3 Chat Template 返回包装对象而不是 Token 列表

自然请求准备阶段曾把 BatchEncoding 直接作为 JSON Prompt，导致不可序列化。

**处理：** 明确提取 `input_ids`，整个失败开发 Run 不计入性能数据；修正后重跑。

### 7.4 原生计数不是总执行账本

新版 `prompt_tokens_by_source{source=local_compute}` 表示初次本地 Prefill，不包含抢占后的全部重复执行。

**处理：** 汇总中分开命名“初次本地 Prefill”和“诊断实际 Prefill”；N0 不安装项目观测时，内部实际执行量标为不可观测，不能填零或声称总重算减少。

### 7.5 观测本身会拖慢调度

只读也需要 CPU 时间。Nobs 两对短跑使 A/640 的 P95 增加 18.58%，C 增加 1.01%。

**处理：** 增加只观测不排序的独立对照；主结果始终比较包含全部候选开销的策略和未修改 N0，不把观测开销随意相减。

## 8. 输出不同怎么判断

先区分状态错误和正常执行次序改变带来的数值差异。旧版丢失生成历史属于真实错误，必须先修复，不能用浮点波动解释。

新版专用正确性运行共 320 请求，192 个基线/候选对输出一致，包括 36 个自然问答对照。但两个 Pilot 中另有 8 个合成请求逐 Token 分歧。对首次分歧位置固定原生历史，读取两个指定 Token 的条件概率。

原先严格的 logprob 差值 1e-3 和 Top-1 一致检查未全部通过；保留失败文件。随后对相关场景补做原生独立重复，原生自身也出现对应近似并列 Token 翻转；所检查 Token 概率最大绝对差约 0.000699。

再做同步/默认、原生/候选、Cold/GPU/CPU 三条缓存路径的长上下文事实检索，共 36/36 回答正确，各配置内跨缓存路径 Token 一致；CPU 路径实际恢复 2048 Tokens / 224 MiB。

这些补充检查支持本次有限范围性能观察，不把事后门槛写成预注册准确率保证，也不声称覆盖全部正式分歧或全词表 KL。`teacher_forced.py`、`quality_followup.py`、`semantic_cache.py`、`assess_quality.py` 保留检查代码。

## 9. 如何避免实验混用

- 原生包与项目扩展隔离，清除旧 PYTHONPATH 和 V1 环境变量，避免新版误加载旧 Overlay。
- 每 Run 新服务，固定预热形状后重置双层缓存，再构造 GPU/CPU 初态；计时前不发会改变初态的探测请求。
- 固定 Trace、模型 Revision、参数和成本表，保存 Hash；失败/被替代的开发版本归档。
- 使用文件锁串行占用 GPU；冻结核心代码，变化后使用新数据目录。断点续跑只跳过已经完整完成的 Run。
- 先验证配对 Trace Hash，再计算每对相对变化，最后对 Run 对求均值及 Bootstrap 区间，不把几千个请求当成几千次独立硬件重复。
- 监护进程在原流水线结束后执行审计和报告；真实失败会记录并停止，不循环重试直到拿到好数字。

## 10. 实际收益与限制

主模型为 Qwen3-1.7B FP16，单 RTX 3090。GPU 640/1024 Blocks 分别约 1.094/1.75 GiB KV，属于受控缓存压力，不代表 24 GiB 卡的自动满显存配置。

| 路径 / 场景 | 对照 | P95 TTFT 变化 | 固定负载输出吞吐变化 |
| --- | --- | ---: | ---: |
| 默认 Async/Graph，A/640 | GPU / N0 | -11.34% | +0.41% |
| 默认 Async/Graph，C/1024 | Tier / N0 | -13.36% | +0.15% |
| 同步 Eager，A/640 | GPU / N0 | -55.67% | +10.52% |
| 同步 Eager，C/1024 | Tier / N0 | -17.61% | +1.49% |
| 同步 Eager，C/1024 | Tier / GPU | -6.31% | 此处不列独立吞吐结论 |

均为 5 对种子，每 Run 256 请求。默认 A 的 P95 区间为 [-15.63%, -7.05%]，C 为 [-14.23%, -12.36%]；同步 Tier/GPU 为 [-10.94%, -2.61%]，均为小样本探索性配对 Bootstrap。

默认 A/C 的冷请求平均 TTFT 增加 14.85%/21.85%；默认 C 吞吐区间跨零。无前缀复用和并发 1 的默认对照没有明确 P95 改善。同步 A/1024 也没有明确 P95 改善。

只改变等待次序可能只是在请求之间转移等待。减少前缀损失、实际计算或搬运，或者改善批处理效率，才可能改善整体吞吐。固定到达速率下，完成速率还受输入限制；当前结果不能宣称默认路径最大容量提高 13.36%。最大 SLO 容量、满显存预算、7B 和新版长测尚未完成。

本轮新版 55 个同步正式 Run、40 个默认正式 Run 于 2026-10-19 21:07（北京时间）完成；包含 Pilot/观测的性能请求共 26112，另有 320 个正确性请求。完成产物检查不等于所有压力输出逐 Token 一致。

## 11. 离线迁移中的问题和处理

引用任务“迁移与回复”对应的离线包主要携带旧版环境。包约 14.23 GB，模型和依赖均为实体文件；目标机从新 Run 开始，不继承源机成功状态。

1. **解释器和二进制依赖不完整：** 单独复制项目和 venv 不能替代底层 Python/Torch 库。构建脚本收集基础 Python 环境、实验 venv、CUDA 工具、Nsight 和独立分析环境，源码在 `offline_migration/build_bundle.py`。
2. **Overlay 链接仍指向源机：** `.so` 等文件最初链接到原安装位置。安装脚本根据新根目录重新建立链接，历史验证修复了 1004 个 Overlay 链接。
3. **模型文件是外部缓存链接：** 构建时实体化模型文件并记录 SHA256。异路径验证检查 9 个模型文件，避免目标离线时依赖源缓存。
4. **Python 环境与脚本保存绝对前缀：** 安装到最终目录后执行 conda-unpack，重建实验 venv，替换项目数据、模型及工具路径；含空格路径也做过验证。
5. **迁移改变了源码字节：** 路径替换后重新冻结，保存原发布 Hash 和迁移修改记录，不把迁移后代码宣称成与源发布字节完全一致。
6. **携带完成标记会跳过目标实验：** 构建时排除 raw 和流水线状态，目标机重新生成结果。
7. **CPU 验证不能冒充 GPU 验证：** 源机异路径验证覆盖导入、模型 Hash、C 编译、策略测试和绘图；目标机需要显式 `verify_offline.py --gpu-smoke`。报告没有把异路径 CPU 检查写成目标服务器性能实测。

宿主 NVIDIA 驱动和系统 glibc 不打包覆盖。当前仓库只发布这些构建/安装/验证代码，不上传环境包或 Release 分卷，也不需要为此配置 Release 上传。

## 12. 代码来源与复现范围

旧版固定 Commit 为 `275de34170654274616082721348b7edd9741d32`，新版 Tag Commit 为 `5df63a7203cdb9328d824a27b682e77c7b9928be`；模型 Revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。

上游修复和策略参考由 `configs/upstream_compat.json`、`upstream_strategy_references.json` 记录，项目本地兼容更正在 `local_compatibility_fixes.json`。参考上游修复与本项目策略分别说明，不将上游原生功能作为项目自研内容。

本次交付保留全部 73 个原始 Python/Shell/Patch 文件及对应配置；依赖、参考成本表和实验方案单独存放。环境路径沿用原实验，部署步骤与需要调整的路径详见 [RUNNING.md](RUNNING.md)。原始性能结果和日志不包含在源码仓库中。

## 13. 本次源码交付前的检查

- 73 个原始 Python/Shell/Patch 文件逐字节比对，均与原项目一致；配置另由 `SOURCE_MANIFEST.json` 校对。
- 从本仓库副本运行 `tests/test_policy.py`：21 passed。
- 从本仓库副本运行 `latest_v0320.test_admission`：3 passed，使用实际原生管理器检查只读查询和排序状态。
- 全部 Python 源码编译检查、Shell `bash -n` 和 JSON 解析通过。
- 本次没有重新加载模型跑 GPU 性能矩阵，表中性能数字来自已完成的原实验。
