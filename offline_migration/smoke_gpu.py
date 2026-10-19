"""Tiny native CPU-offload + optimized-engine check on the target GPU."""
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    import torch
    assert torch.cuda.is_available(), 'NVIDIA driver/GPU is unavailable.'
    gpu = torch.cuda.get_device_properties(0)
    assert gpu.major >= 8, 'This experiment uses an Ampere-or-newer attention path.'
    os.environ['KV_SCHED_CONFIG'] = json.dumps(dict(admission='tier', preemption='restore',
        cost_model=str(root/'data/cost_models/qwen3-1.7b-3090.json')))
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    llm = LLM(model=str(root/'models/Qwen3-1.7B'), dtype='float16', enforce_eager=True,
        max_model_len=8192, block_size=16, num_gpu_blocks_override=640,
        gpu_memory_utilization=.8, max_num_batched_tokens=1024, max_num_seqs=4,
        enable_prefix_caching=True, enable_chunked_prefill=True, async_scheduling=False,
        disable_log_stats=True, kv_transfer_config=KVTransferConfig(
            kv_connector='OffloadingConnector', kv_role='kv_both',
            kv_connector_extra_config={'num_cpu_blocks':32, 'block_size':256,'eviction_policy':'lru'}))
    outputs = llm.generate(['请简短介绍 KV Cache。', 'Write a short Python hello world.'],
                          SamplingParams(temperature=0, max_tokens=16))
    assert len(outputs) == 2 and all(x.finished and x.outputs[0].token_ids for x in outputs)
    result = dict(passed=True, gpu=gpu.name, memory_bytes=gpu.total_memory,
                  requests=2, output_tokens=[len(x.outputs[0].token_ids) for x in outputs])
    (root/'data/offline_gpu_smoke.json').write_text(json.dumps(result, indent=2))
    print('OFFLINE_GPU_VERIFIED', result, flush=True)
    llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
