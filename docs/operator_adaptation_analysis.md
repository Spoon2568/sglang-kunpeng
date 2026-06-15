# Kunpeng 算子适配分析：KunPengDistInfer vs SGLang

## 一、已适配算子（✅ 完成）

| 算子类别 | C++ 算子 | SGLang 集成位置 | 测试状态 | 备注 |
|---------|---------|----------------|---------|------|
| **Dense Linear INT8** | `igemm_pack_weight_out`, `igemm_pack_act_out`, `igemm_bdq_out` | `W8A8Int8LinearMethod.apply` | ✅ 需端到端测试 | FFN/MLP 层 |
| **MoE INT8** | `igemm_fusedmoe_gateup_out`, `igemm_fusedmoe_down_out` | `W8A8Int8MoEMethod.apply` | ✅ 需端到端测试 | 融合 silu+mul |
| **MLA INT8 BMM** | `batched_gemm_woqs8_allthreads_out`, `batched_gemm_pack_allthreads_out` | `forward_mla.py` (q_nope×w_kc, attn×w_vc) | ✅ 需端到端测试 | DeepSeek MLA 核心 |
| **RMSNorm** | `rmsnorm_out`, `add_rmsnorm_out` | `RMSNorm._forward_method` | ✅ 4/4 单元测试通过 | 支持 residual |
| **Grouped TopK** | `grouped_topk_out` | `grouped_topk_cpu`, `biased_grouped_topk_cpu` | ✅ 4/4 单元测试通过 | MoE 路由 |
| **Router BF16 GEMM** | `bgemm_out`, `bgemm_pack_out` | `MoEGate.forward` | ✅ 3/3 单元测试通过 | MoE gate logits |
| **Embedding** | `embedding_out` | `UnquantizedEmbeddingMethod.embedding` | ✅ 3/3 单元测试通过 | 支持 TP 切分 |
| **MulScalarAdd** | `mul_scalar_add_out` | `DeepseekV2MoE.forward_normal` | ✅ 7/7 单元测试通过 | MoE shared+routed 合并 |
| **Argmax** | `argmax_out`, `argmax_merge_out` | `sampler.py` greedy path | ✅ 7/7 单元测试通过 | Greedy sampling，仅单机 |
| **Element-wise (通用)** | `add_out`, `mul_out`, `tanh_out`, `add_scalar_out`, `tanh_backward_out` | 工具函数 `kunpeng/elementwise.py` | ✅ 通过 mul_scalar_add 验证 | 按需调用 |

**总计**：10 组算子，28 个单元测试全部通过 ✅

---

## 二、未适配但 C++ 库已提供的算子

### 2.1 **量化相关**（高优先级）

| 算子 | C++ 函数 | KunPengDistInfer 用途 | SGLang 对应位置 | 适配难度 | 优先级 |
|------|---------|---------------------|---------------|---------|-------|
| **act_scale_pack** | `act_scale_pack_out` | 激活值量化前的 scale 打包 | `W8A8Int8LinearMethod.apply` 内部 | 低（已有 INT8 路径） | 🟡 中 |
| **quant** | `quant_out` | BF16 → INT8 量化 | 各层前向传播内部 | 低（已集成） | 🟢 低 |
| **quant_pack** | `quant_pack_out` | 量化后打包（用于 igemm） | 各层前向传播内部 | 低（已集成） | 🟢 低 |
| **silu_mul_quant** | `silu_mul_quant_out` | MLP: silu+mul+quant 融合 | `DeepseekV2MLP` | 中（需验证融合收益） | 🟡 中 |
| **moe_silu_mul_quant** | `moe_silu_mul_quant_out` | MoE: silu+mul+quant 融合 | `DeepseekV2MoE` | 中（已有 INT8 MoE） | 🟢 低 |
| **add_rmsnorm_quant** | `add_rmsnorm_quant_out` | residual add + rmsnorm + quant 融合 | `RMSNorm.forward` | 中（三算子融合） | 🟡 中 |
| **rmsnorm_quant** | `rmsnorm_quant_out` | rmsnorm + quant 融合 | `RMSNorm.forward` | 低（二算子融合） | 🟡 中 |

**分析**：这些是**融合算子**，减少中间 tensor 传输。当前 SGLang 已有 INT8 路径，但未做融合优化。建议**先端到端测试现有 INT8 算子性能**，再决定是否融合。

---

### 2.2 **RoPE/位置编码**（高优先级）

| 算子 | C++ 函数 | KunPengDistInfer 用途 | SGLang 对应位置 | 适配难度 | 优先级 |
|------|---------|---------------------|---------------|---------|-------|
| **yarn** | `yarn_out` | DeepSeek YaRN RoPE（扩展上下文） | `deepseek_common/attention_forward_methods/forward_mla.py` | 中（需理解 YaRN 算法） | 🔴 高 |
| **yarn_init_cache** | `yarn_init_cache_out` | YaRN cache 初始化 | MLA 初始化阶段 | 中 | 🟡 中 |
| **rope_forward_impl** | `rope_forward_impl` | 标准 RoPE | `rotary_embedding.py` | 低（标准 RoPE） | 🟡 中 |
| **rope_backward_impl** | `rope_backward_impl` | RoPE 反向传播 | 训练场景 | 低 | 🟢 低（推理不需要） |

**分析**：DeepSeek 使用 **YaRN RoPE**（支持长上下文），当前 SGLang 可能用 PyTorch 实现。**yarn 是高优先级**，因为 RoPE 是 attention 热点。

---

### 2.3 **KV Cache 管理**（高优先级）

| 算子 | C++ 函数 | KunPengDistInfer 用途 | SGLang 对应位置 | 适配难度 | 优先级 |
|------|---------|---------------------|---------------|---------|-------|
| **concat_and_cache_mla** | `concat_and_cache_mla_out` | MLA: concat KV 并写入 cache | `forward_mla.py` | 高（MLA 特有格式） | 🔴 高 |
| **swap_concat_and_cache_mla** | `swap_concat_and_cache_mla_out` | MLA: swap 模式（跨节点？） | `forward_mla.py` | 高 | 🟡 中 |
| **concat_and_cache_kv** | `concat_and_cache_kv_out` | 标准 KV cache concat | `attention/` | 中 | 🟡 中 |
| **swap_concat_and_cache_kv** | `swap_concat_and_cache_kv_out` | 标准 KV swap | `attention/` | 中 | 🟢 低 |

**分析**：MLA 的 KV cache 格式特殊（compressed KV），`concat_and_cache_mla` 是**核心热点**。SGLang 当前可能用 FlashInfer 的 cache 管理。

---

### 2.4 **Attention 核心**（最高优先级）

| 算子 | C++ 函数 | KunPengDistInfer 用途 | SGLang 对应位置 | 适配难度 | 优先级 |
|------|---------|---------------------|---------------|---------|-------|
| **flash_mla_with_kvcache** | `flash_mla_with_kvcache_out` | **MLA flash attention (prefill + decode)** | `forward_mla.py` | 🔥 极高（MLA 核心） | 🔴 **最高** |
| **flash_mla_get_metadata** | `flash_mla_get_metadata_out` | MLA metadata 计算 | `forward_mla.py` | 高 | 🔴 高 |
| **varlen_attention** | `varlen_attention_out` | 标准 varlen attention (prefill) | `attention/` | 高 | 🔴 高 |
| **varlen_attention_gqa** | `varlen_attention_gqa_out` | GQA varlen attention | `attention/` | 高 | 🟡 中 |

**分析**：
- **`flash_mla_with_kvcache`** 是 **DeepSeek MLA 的核心算子**，融合了 QK^T、softmax、attention output 计算。SGLang 当前可能用 FlashInfer 的标准 attention，**不支持 MLA 压缩格式**。
- **这是性能瓶颈所在**，必须适配才能发挥 kunpeng 优势。

---

### 2.5 **分布式通信**（TP 必需）

| 算子 | C++ 函数 | KunPengDistInfer 用途 | SGLang 对应位置 | 适配难度 | 优先级 |
|------|---------|---------------------|---------------|---------|-------|
| **all_reduce_no_copy_inout** | （通信算子） | TP allreduce（FFN/MLP 输出） | `parallel_state.py` | 高（需多卡环境） | 🔴 高（TP>1） |
| **allgather** | （通信算子） | TP allgather（router logits、argmax） | `parallel_state.py` | 高 | 🔴 高（TP>1） |
| **allgather_v2** | （通信算子） | allgather 优化版本 | `parallel_state.py` | 高 | 🟡 中 |
| **allgather_comm8** | （通信算子） | INT8 压缩通信 | `parallel_state.py` | 高 | 🟡 中 |
| **mla_q_alltoall** | （通信算子） | MLA Q 矩阵 alltoall（TP） | `forward_mla.py` | 极高 | 🔴 高（TP>1） |
| **mla_o_alltoall** | （通信算子） | MLA O 矩阵 alltoall（TP） | `forward_mla.py` | 极高 | 🔴 高（TP>1） |
| **mla_alltoall_fence** | （通信算子） | alltoall fence（同步） | `forward_mla.py` | 中 | 🟡 中 |
| **barrier** | （通信算子） | 同步屏障 | 各通信点 | 低 | 🟢 低 |
| **global_barrier** | （通信算子） | 全局屏障 | 初始化 | 低 | 🟢 低 |

**分析**：
- 当前 SGLang 用 `torch.distributed` 或 vLLM 的通信层。
- **MLA 的 alltoall** 是特殊的（Q/O 矩阵重排），需要深度集成。
- **TP > 1 时必须适配**，否则无法多卡推理。

---

### 2.6 **工具/辅助算子**（低优先级）

| 算子 | C++ 函数 | KunPengDistInfer 用途 | SGLang 对应位置 | 适配难度 | 优先级 |
|------|---------|---------------------|---------------|---------|-------|
| **tcopy** | `tcopy_out` | Tensor copy（transpose、slice） | 各处 | 低 | 🟢 低 |
| **zero** | `zero_out` | Tensor 清零 | 初始化 | 低 | 🟢 低 |
| **silu_out** | `silu_out` | SiLU 激活（未融合版本） | `DeepseekV2MLP` | 低（已有融合版） | 🟢 低 |
| **silu_backward_out** | `silu_backward_out` | SiLU 反向 | 训练 | 低 | 🟢 低（推理不需要） |
| **rmsnorm_backward_out** | `rmsnorm_backward_out` | RMSNorm 反向 | 训练 | 低 | 🟢 低（推理不需要） |
| **update_context_interlayer** | （未在 C++ 找到） | 更新层间上下文？ | — | 未知 | 🟢 低 |

**分析**：这些是辅助算子，优先级低，PyTorch 原生实现足够。

---

### 2.7 **未在 C++ 库找到的算子**

| 算子 | KunPengDistInfer 调用 | 状态 | 备注 |
|------|---------------------|------|------|
| **dispatch_send/recv** | MoE dispatch 通信 | ❓ 未找到 | 可能在通信层实现 |
| **combine_send/recv** | MoE combine 通信 | ❓ 未找到 | 可能在通信层实现 |

---

## 三、适配优先级建议

### 🔴 **P0 - 最高优先级（性能关键 + TP 必需）**

1. **flash_mla_with_kvcache** — DeepSeek MLA attention 核心，prefill + decode 的瓶颈
2. **yarn** — DeepSeek YaRN RoPE，长上下文支持
3. **concat_and_cache_mla** — MLA KV cache 管理
4. **all_reduce / allgather** — TP > 1 必需（FFN/MLP allreduce，router allgather）
5. **mla_q_alltoall / mla_o_alltoall** — MLA TP 通信

**理由**：这些是 **DeepSeek-R1 MLA + TP 推理的核心路径**，不适配则性能提升有限或无法多卡。

---

### 🟡 **P1 - 高优先级（性能优化）**

6. **silu_mul_quant / moe_silu_mul_quant** — 融合算子，减少中间 tensor
7. **add_rmsnorm_quant / rmsnorm_quant** — 融合 normalization + quantization
8. **varlen_attention / varlen_attention_gqa** — 标准 attention 优化（非 MLA 模型）
9. **allgather_comm8** — INT8 压缩通信（带宽优化）

**理由**：这些是**性能优化**，可以在 P0 完成后逐步添加。

---

### 🟢 **P2 - 低优先级（边缘场景/训练）**

10. 训练相关：`rope_backward`, `silu_backward`, `rmsnorm_backward`
11. 辅助工具：`tcopy`, `zero`, `silu_out`（未融合版本）
12. 非 MLA cache：`concat_and_cache_kv`, `swap_concat_and_cache_kv`

**理由**：推理场景不需要，或 PyTorch 原生实现足够。

---

## 四、当前架构差异分析

### 4.1 **SGLang vs KunPengDistInfer 架构对比**

| 维度 | SGLang | KunPengDistInfer | 差异影响 |
|------|--------|-----------------|---------|
| **Attention 后端** | FlashInfer（标准 attention） | 自定义 flash_mla_with_kvcache（MLA 压缩） | 🔴 **MLA 无法使用 FlashInfer** |
| **KV Cache 格式** | 标准 KV（per-head） | MLA 压缩格式（latent KV） | 🔴 Cache 管理需重写 |
| **RoPE 实现** | PyTorch/triton | YaRN C++ 算子 | 🟡 YaRN 长上下文需适配 |
| **TP 通信** | torch.distributed / vLLM | 自定义通信算子 + MLA alltoall | 🔴 MLA alltoall 需深度集成 |
| **量化路径** | 动态量化 per-token | 静态 scale（预计算） | 🟡 当前 INT8 已适配，融合优化可后续 |
| **MoE Dispatch** | 标准 dispatch + combine | 可能有自定义通信（未确认） | 🟢 当前 INT8 MoE 已工作 |

---

### 4.2 **关键难点：MLA Attention 适配**

**问题**：
- SGLang 的 `forward_mla.py` 用 **FlashInfer** 做 attention，但 FlashInfer 不理解 MLA 的 compressed KV 格式。
- KunPengDistInfer 的 `flash_mla_with_kvcache` 是**端到端融合算子**：
  1. Q × compressed_KV → expanded_KV
  2. FlashAttention(Q, expanded_KV)
  3. 写回 compressed_KV cache

**适配路径**：
1. **替换 forward_mla.py 的 FlashInfer 调用** → 改为 `flash_mla_with_kvcache`
2. **修改 KV cache 管理** → 使用 `concat_and_cache_mla`
3. **TP 场景** → 加入 `mla_q_alltoall` / `mla_o_alltoall`

**难度**：🔥 **极高**（需深入理解 MLA 算法 + SGLang attention 架构）

---

## 五、推荐实施路线图

### **阶段 1：验证当前适配（1-2 天）**

- ✅ 已完成单元测试（28/28 通过）
- 🔲 **端到端推理测试**：
  - 单卡 TP=1，DeepSeek-R1-Channel-INT8
  - 验证吞吐量、精度、显存占用
  - **检查是否真正走了 kunpeng 算子**（添加 profiling）

**目标**：确认当前 INT8 linear/MoE/norm/topk 是否生效，测量基线性能。

---

### **阶段 2：MLA Attention 适配（1-2 周）**

1. **学习 MLA 算法**：
   - 阅读 DeepSeek-V2 论文（MLA 部分）
   - 理解 compressed KV 格式

2. **实现 flash_mla_with_kvcache 封装**：
   - 创建 `kunpeng/attention.py`
   - 封装 C++ `flash_mla_with_kvcache_out`
   - 处理 metadata 计算

3. **集成到 forward_mla.py**：
   - 替换 FlashInfer 调用
   - 修改 KV cache 管理逻辑
   - 单卡验证正确性

4. **测试**：
   - 对比 PyTorch 参考实现
   - Prefill + decode 场景
   - 不同 seq_len

**目标**：MLA attention 走 kunpeng 算子，性能提升显著。

---

### **阶段 3：YaRN RoPE 适配（3-5 天）**

1. **实现 yarn 封装**：
   - `kunpeng/rope.py`
   - 封装 `yarn_out`

2. **集成到 forward_mla.py**：
   - 替换 RoPE 计算

3. **测试长上下文**：
   - 32k / 64k / 128k context

**目标**：长上下文性能优化。

---

### **阶段 4：TP 通信适配（1-2 周，需多卡环境）**

1. **实现标准通信算子**：
   - `kunpeng/communicator.py`
   - 封装 `all_reduce`, `allgather`

2. **实现 MLA alltoall**：
   - 封装 `mla_q_alltoall`, `mla_o_alltoall`
   - 集成到 forward_mla.py

3. **TP=2/4/8 测试**：
   - 验证正确性
   - 测量通信开销

**目标**：TP > 1 可用，多卡加速。

---

### **阶段 5：融合算子优化（可选，1 周）**

1. **add_rmsnorm_quant / rmsnorm_quant**
2. **silu_mul_quant**
3. **allgather_comm8**（INT8 压缩通信）

**目标**：进一步性能提升 5-10%。

---

## 六、总结

### 已适配（10 组）
✅ Dense Linear INT8, MoE INT8, MLA INT8 BMM, RMSNorm, Grouped TopK, Router GEMM, Embedding, MulScalarAdd, Argmax, Element-wise

### 未适配但高优先级（5 组）
🔴 **flash_mla_with_kvcache**, **yarn**, **concat_and_cache_mla**, **all_reduce/allgather**, **mla_alltoall**

### 关键瓶颈
1. **MLA Attention**：当前用 FlashInfer（不支持 MLA），必须适配 `flash_mla_with_kvcache`
2. **TP 通信**：TP > 1 时需要通信算子 + MLA alltoall
3. **YaRN RoPE**：长上下文支持

### 建议
**先端到端测试当前适配的性能**，再决定是否投入 MLA attention 的重构（工作量大，但性能提升预期显著）。
