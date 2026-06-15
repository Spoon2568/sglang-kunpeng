# DeepSeek-R1-Channel-INT8 Kunpeng 适配 - 剩余算子清单

## 当前状态概览

### ✅ 已适配（10 组，28 个测试通过）
1. Dense Linear INT8 (`igemm_bdq`, `igemm_pack`)
2. MoE INT8 (`igemm_fusedmoe_gateup/down`)
3. MLA INT8 BMM (`batched_gemm_woqs8_allthreads`)
4. RMSNorm (`rmsnorm_out`, `add_rmsnorm_out`)
5. Grouped TopK (`grouped_topk_out`)
6. Router BF16 GEMM (`bgemm_out`)
7. Embedding (`embedding_out`)
8. MulScalarAdd (`mul_scalar_add_out`)
9. Argmax (`argmax_out`)
10. TP 通信 (`KunpengCommunicator`: allreduce/allgather/barrier)

### ❌ 未适配但必需的算子

---

## 一、🔴 P0 - 最高优先级（性能瓶颈）

### 1. **MLA Flash Attention** 🔥

**C++ 算子**：
- `flash_mla_with_kvcache_out(q, kv_cache, kpe_cache, output, ...)`
- `flash_mla_get_metadata_out(...)`

**当前 SGLang 实现**：
- `forward_mla.py` line 467-487: 使用 **FlashInfer** 的标准 attention
- FlashInfer 不理解 MLA 的 compressed KV 格式

**适配难度**：🔥 极高
- 需要理解 MLA 算法（compressed KV、latent attention）
- 替换 FlashInfer 调用
- 修改 KV cache 管理逻辑

**性能影响**：⭐⭐⭐⭐⭐
- Attention 占推理时间 **50-70%**
- 这是 **最大性能瓶颈**

**优先级**：🔴 **最高**

---

### 2. **YaRN RoPE**

**C++ 算子**：
- `yarn_out(input, output, position_ids, cos_sin_cache, ...)`
- `yarn_init_cache_out(freqs, cache, ...)`

**当前 SGLang 实现**：
- `rope_variant.py` line 335-395: `YarnRotaryEmbedding` 类（PyTorch 实现）
- DeepSeek config: `rope_scaling["rope_type"] = "deepseek_yarn"`（line 1454）
- `forward_mla.py` line 382-487: RoPE 应用在 q_pe/k_pe 上

**适配难度**：🟡 中等
- YaRN 算法相对标准（论文公开）
- 需要封装 `yarn_out` 并集成到 `forward_mla.py`

**性能影响**：⭐⭐⭐
- RoPE 计算量不大，但**长上下文场景**（32k/64k/128k）会成为瓶颈

**优先级**：🔴 高

---

### 3. **MLA KV Cache 管理**

**C++ 算子**：
- `concat_and_cache_mla_out(kv_new, kpe_new, kv_cache, kpe_cache, ...)`
- `swap_concat_and_cache_mla_out(...)` — swap 模式（跨节点？）

**当前 SGLang 实现**：
- `forward_mla.py` 使用 FlashInfer 的标准 cache 管理
- MLA 的 cache 格式特殊：`[compressed_kv, kpe]`，不是标准的 `[k, v]`

**适配难度**：🟡 中等
- 需要配合 `flash_mla_with_kvcache` 一起适配
- 理解 MLA 的 latent KV + rope KV 拆分

**性能影响**：⭐⭐⭐
- Cache 写入不是瓶颈，但**格式错误会导致 attention 失败**

**优先级**：🔴 高（与 flash_mla_with_kvcache 绑定）

---

## 二、🟡 P1 - 高优先级（性能优化）

### 4. **MLA AllToAll 通信（TP 场景）**

**C++ 算子**：
- `mla_q_alltoall(q_local, q_global, ...)` — Q 矩阵重排
- `mla_o_alltoall(o_local, o_global, ...)` — O 矩阵重排
- `mla_alltoall_fence(...)` — 同步 fence

**当前 SGLang 实现**：
- `forward_mla.py` 没有 MLA alltoall（可能用标准 TP 或未支持）

**适配难度**：🔥 高
- MLA 在 TP 场景下需要**特殊的 Q/O 矩阵重排**
- 不同于标准 attention 的 allreduce
- 需要理解 MLA 的 TP 切分策略

**性能影响**：⭐⭐⭐⭐
- **仅 TP > 1 时需要**
- 如果不适配，MLA 在多卡场景下可能无法正确工作或性能极差

**优先级**：🟡 高（TP=1 可跳过）

**是否需要**：
- TP=1（单卡）→ ❌ 不需要
- TP=2/4/8（多卡）→ ✅ **必需**

---

### 5. **融合量化算子（减少中间 tensor）**

#### 5.1 `add_rmsnorm_quant_out`

**语义**：`residual_add + rmsnorm + quant` 三算子融合

**当前 SGLang**：
- `RMSNorm.forward` → `add_rmsnorm_out`（已适配）
- 量化在后续单独做

**适配难度**：🟡 中
- 需要在 `RMSNorm` 里加融合路径
- 输出 INT8 而非 BF16

**性能影响**：⭐⭐
- 减少一次 BF16 tensor 写回
- 预期提升 **5-10%**

**优先级**：🟡 中

---

#### 5.2 `rmsnorm_quant_out`

**语义**：`rmsnorm + quant` 二算子融合

**当前 SGLang**：同上

**适配难度**：🟢 低
- 比 `add_rmsnorm_quant` 简单（少一个 residual add）

**性能影响**：⭐⭐

**优先级**：🟡 中

---

#### 5.3 `silu_mul_quant_out`

**语义**：`silu(x) * y + quant` 融合（MLP gateup 后）

**当前 SGLang**：
- `DeepseekV2MLP.forward` → 先 `igemm_fusedmoe_gateup`（已融合 silu*mul）
- 量化在后续单独做

**适配难度**：🟡 中
- MLP 路径需要加融合量化

**性能影响**：⭐⭐

**优先级**：🟡 中

---

#### 5.4 `moe_silu_mul_quant_out`

**语义**：MoE 版本的 `silu_mul_quant`

**当前 SGLang**：
- `DeepseekV2MoE` → 已有 `igemm_fusedmoe_gateup`（融合 silu*mul）

**适配难度**：🟡 中

**性能影响**：⭐⭐

**优先级**：🟡 中

---

### 6. **标准 Attention（非 MLA 模型）**

**C++ 算子**：
- `varlen_attention_out(q, k, v, output, ...)` — prefill
- `varlen_attention_gqa_out(...)` — GQA variant

**当前 SGLang**：
- 使用 FlashInfer 或 xFormers

**适配难度**：🟡 中

**性能影响**：⭐⭐⭐
- **仅非 MLA 模型需要**（如 Llama、Qwen 等）
- DeepSeek-R1 用 MLA，不需要这个

**优先级**：🟢 低（DeepSeek-R1 不需要）

---

## 三、🟢 P2 - 低优先级（边缘场景）

### 7. **标准 KV Cache 管理**

**C++ 算子**：
- `concat_and_cache_kv_out(...)` — 标准 KV cache
- `swap_concat_and_cache_kv_out(...)` — swap 模式

**优先级**：🟢 低（DeepSeek-R1 用 MLA cache）

---

### 8. **工具算子**

**C++ 算子**：
- `tcopy_out(...)` — tensor copy/transpose
- `zero_out(...)` — tensor 清零
- `silu_out(...)` — SiLU（未融合版本）

**优先级**：🟢 低（PyTorch 原生实现足够）

---

### 9. **训练相关**

**C++ 算子**：
- `rope_backward_impl(...)`
- `silu_backward_out(...)`
- `rmsnorm_backward_out(...)`

**优先级**：🟢 极低（推理不需要）

---

## 四、实施优先级总结

### 🔴 **必须适配（P0）**

| 算子 | 难度 | 性能影响 | TP=1 是否需要 | TP>1 是否需要 |
|------|------|----------|-------------|-------------|
| **flash_mla_with_kvcache** | 🔥 极高 | ⭐⭐⭐⭐⭐ | ✅ 必需 | ✅ 必需 |
| **yarn** | 🟡 中 | ⭐⭐⭐ | ✅ 必需 | ✅ 必需 |
| **concat_and_cache_mla** | 🟡 中 | ⭐⭐⭐ | ✅ 必需 | ✅ 必需 |

**这 3 个算子是 DeepSeek-R1 MLA 推理的核心**，不适配则：
- Attention 仍走 PyTorch/FlashInfer（无法发挥 kunpeng 优势）
- 长上下文性能差（YaRN RoPE 未优化）
- KV cache 格式不对（可能导致错误）

---

### 🟡 **推荐适配（P1）**

| 算子 | 难度 | 性能影响 | 适用场景 |
|------|------|----------|---------|
| **mla_q/o_alltoall** | 🔥 高 | ⭐⭐⭐⭐ | TP > 1 |
| **add_rmsnorm_quant** | 🟡 中 | ⭐⭐ | 所有场景 |
| **silu_mul_quant** | 🟡 中 | ⭐⭐ | 所有场景 |

**预期收益**：
- MLA alltoall：TP > 1 时必需，否则多卡无法正确工作
- 融合量化：5-10% 性能提升

---

### 🟢 **可选适配（P2）**

- 标准 attention（非 MLA 模型）
- 标准 KV cache
- 工具算子、训练算子

---

## 五、实施建议

### **阶段 1：验证当前适配（1 天）**

目标：确认已适配的 10 组算子是否真正生效

**任务**：
1. 端到端推理测试（DeepSeek-R1-Channel-INT8, TP=1）
2. Profiling：确认 `igemm_bdq`, `igemm_fusedmoe`, `rmsnorm_out` 等是否被调用
3. 测量吞吐量、延迟、显存

**判断标准**：
- 如果性能提升明显（如吞吐量 +30%）→ 说明 linear/MoE/norm 算子生效
- 如果提升不明显（<10%）→ 说明 attention 是瓶颈，必须做阶段 2

---

### **阶段 2：MLA Attention 适配（1-2 周）** 🔥

目标：适配 P0 三大算子

**任务**：
1. **学习 MLA 算法**（2-3 天）
   - 阅读 DeepSeek-V2 论文
   - 理解 compressed KV、latent attention
   - 分析 `flash_mla_with_kvcache` C++ 实现

2. **封装 flash_mla_with_kvcache**（3-4 天）
   - 创建 `kunpeng/attention.py`
   - 封装 C++ 算子
   - 处理 metadata

3. **封装 yarn + concat_and_cache_mla**（2-3 天）
   - 创建 `kunpeng/rope.py` 和 `kunpeng/kvcache.py`
   - 封装 C++ 算子

4. **集成到 forward_mla.py**（3-4 天）
   - 替换 FlashInfer 调用（line 467-487）
   - 替换 YarnRotaryEmbedding（line 382-425）
   - 修改 cache 管理逻辑
   - 单卡 TP=1 验证

5. **测试**（2-3 天）
   - 正确性：对比 PyTorch 参考
   - 性能：prefill + decode 吞吐量
   - 长上下文：32k/64k/128k

**预期收益**：
- Attention 加速 **2-3x**
- 端到端吞吐量 **+50-100%**

---

### **阶段 3：MLA AllToAll（TP > 1 场景，1 周）**

**前提**：阶段 2 完成，且需要多卡推理

**任务**：
1. 封装 `mla_q/o_alltoall`（2-3 天）
2. 集成到 `forward_mla.py`（2-3 天）
3. TP=2/4/8 验证（2 天）

**预期收益**：
- TP > 1 可用
- 线性扩展（理想情况）

---

### **阶段 4：融合量化优化（可选，1 周）**

**前提**：阶段 2/3 完成，性能已有大幅提升

**任务**：
1. 适配 `add_rmsnorm_quant`, `silu_mul_quant`
2. 集成到 `RMSNorm`, `DeepseekV2MLP/MoE`
3. 端到端测试

**预期收益**：
- 额外 **5-10%** 性能提升

---

## 六、总结

### 必须适配的算子（3 个）

1. **flash_mla_with_kvcache** 🔥 — MLA attention 核心
2. **yarn** — YaRN RoPE
3. **concat_and_cache_mla** — MLA KV cache

### 推荐适配的算子（3 个）

4. **mla_q/o_alltoall** — MLA TP 通信（TP > 1 必需）
5. **add_rmsnorm_quant** — 融合量化
6. **silu_mul_quant** — 融合量化

### 关键结论

- **当前已适配 10 组算子**，但 **attention 仍走 PyTorch/FlashInfer**
- **MLA attention 是最大瓶颈**（占 50-70% 时间），不适配则性能提升有限
- **建议先端到端测试**（验证 linear/MoE 是否生效），再决定是否投入 MLA attention 重构
- **MLA attention 适配难度极高**（1-2 周），但预期性能提升显著（+50-100%）

---

## 附录：快速检查清单

用这个命令检查当前推理是否走了 kunpeng 算子：

```bash
# 添加 debug 日志
export SGLANG_KUNPENG_DEBUG=1

# 运行推理
python -m sglang.launch_server ... 2>&1 | grep -E "kunpeng|igemm|flash_mla"
```

如果看到 `igemm_bdq`, `igemm_fusedmoe` 等日志 → 说明 linear/MoE 已生效  
如果没看到 `flash_mla_with_kvcache` → 说明 attention 仍走 PyTorch
