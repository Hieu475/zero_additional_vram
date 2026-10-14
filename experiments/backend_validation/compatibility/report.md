# Backend compatibility audit

Model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`. GPU: `NVIDIA GeForce RTX 4050 Laptop GPU` (compute capability `(8, 9)`).
Software: PyTorch `2.11.0+cu128`, CUDA `12.8`, Transformers `5.14.1`.

## Local probe

Model config: `{'available_in_local_hf_cache': True, 'model_type': 'qwen2', 'architectures': ['Qwen2ForCausalLM'], 'num_hidden_layers': 36, 'hidden_size': 2048}`.
Current ZASSD APIs: `{'qwen_layer_path_supported': True, 'layer_skip_context_manager_present': True, 'custom_cache_accepts_hf_cache_object': True, 'speculative_code_passes_custom_cache': True}`.

| Candidate | Local state | RTX 4050 / Qwen support | Layer skipping + custom verification |
|---|---|---|---|
| HF + bitsandbytes NF4 | tested_baseline | confirmed_by_existing_runs | confirmed_in_current_code |
| HF + HQQ W4A16 + GemLite | smoke_passed_via_manual_linear_patch | RTX_4050_Qwen2.5-3B_smoke_passed | PLD_K4_and_CKA90_K2_static_KV_smoke_passed |
| HF + HQQ torchao_int4 fused kernel | dependencies_missing | Ada_tested_by_HQQ_adapter_source | architecture_should_be_preserved; runtime_unverified |
| HF + torch.compile on BNB NF4 | compile_not_attempted | same_model_and_gpu_as_baseline | dynamic_control_flow_and_cache_recompile_risk |
| vLLM | installed | package_present_but_model_and_6GB_smoke_unverified | custom_layer_skip_and_cache_intervention_not_established |
| TensorRT-LLM | not_installed | not_locally_verified | requires_engine_level_integration |

## Decision

HQQ+GemLite passes the one-prompt runtime compatibility gate through its manual Transformers linear patch: Qwen2.5-3B loaded on RTX 4050, 36-layer access remained intact, static KV vanilla ran, and PLD K4 plus CKA90/K2 completed with exact token agreement against this backend's vanilla output. The built-in Transformers HQQ quantization_config route is blocked by an unimplemented get_quantize_ops() in Transformers 5.14.1; preserve the manual patch path and verify across more prompts before performance integration.

Runtime smoke record: `{'artifact': 'experiments/backend_validation/compatibility/hqq_gemlite_smoke.json', 'sha256': 'aa1ef769d7dc1dcdee8619ce04c0a17d687bbf534e6e280aad51a622539d9fa8', 'passed': True, 'loader_path': 'manual_hqq_gemlite_linear_patch', 'hqq_version': '0.2.8.post1', 'gemlite_version': '0.6.0.post2', 'vanilla_static_kv_passed': True, 'pld_k4_ran_and_exact': True, 'cka90_k2_ran_and_exact': True, 'peak_allocated_mb': 2489.51123046875}`.

HQQ+GemLite is the leading candidate because it replaces linear layers while preserving the Hugging Face model tree, intermediate layer list and cache API. Official documentation covers HQQ 4-bit FP16 at axis=1 and GemLite Triton kernels, while the local smoke supplies the RTX 4050/Qwen2.5-3B evidence.
HQQ's torchao fused INT4 adapter has explicit Ada testing but requires BF16 compute, so it is a useful alternative rather than a same-precision comparison.
The smoke is a compatibility result only. Its first-use latency includes JIT/kernel setup and CPU-side quantization; do not use it as a paired performance result.

## Primary sources

- [hf_hqq](https://huggingface.co/docs/transformers/v5.13.0/en/quantization/hqq)
- [hqq_readme](https://github.com/dropbox/hqq/blob/master/Readme.md)
- [hqq_torchao_adapter](https://github.com/dropbox/hqq/blob/master/hqq/backends/torchao.py)
- [gemlite_readme](https://github.com/dropbox/gemlite)
- [torchao_inference](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- [torchao_hqq_proto](https://github.com/pytorch/ao/tree/main/torchao/prototype/hqq)
- [vllm_quantization](https://docs.vllm.ai/en/stable/features/quantization/)
