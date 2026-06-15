# Kunpeng 其他算子 Code Review

**Review 时间**: 2026-06-13  
**Review 范围**: MoE INT8、MLA INT8 BMM、RMSNorm

---

## 📊 MoE INT8 Review

### 实现路径
```
W8A8Int8MoEMethod.apply()
  └─> apply_moe() (kunpeng/quantization/w8a8_int8.py:112)
        ├─> torch.ops.async_compute.igemm_fusedmoe_gateup_out()
        └─> torch.ops.async_compute.igemm_fusedmoe_down_out()
```

### 核心逻辑

1. **Token 路由**
   ```python
   sorted_token_ids, sorted_weights, experts_offset = sort_tokens_by_expert(
       topk_ids, topk_weights, num_experts
   )
   ```

2. **Gate-Up Fusion (INT8 GEMM)**
   ```python
   # x_q: [num_tokens, hidden] INT8
   # w13: [num_experts, 2*intermediate, hidden] INT8
   # → gateup_out: [routed_tokens, 2*intermediate] BF16
   torch.ops.async_compute.igemm_fusedmoe_gateup_out(
       acts_and_scale, w13_weight, w13_weight_scale, sorted_token_ids,
       experts_offset, gateup_out, ...
   )
   ```

3. **SiLU + Down Projection**
   ```python
   # silu(gateup[:, :intermediate]) * gateup[:, intermediate:]
   act_q, act_scale = quantize_int8(gated_up)
   
   # down: [routed_tokens, hidden] BF16
   torch.ops.async_compute.igemm_fusedmoe_down_out(
       act_q, w2_weight, act_scale, w2_weight_scale, ...
   )
   ```

4. **Combine (weighted scatter-add)**
   ```python
   down_out = down_out * sorted_weights.view(-1, 1)
   output = torch.zeros((num_tokens, hidden), dtype=bf16)
   output.index_add_(0, sorted_token_ids, down_out)
   ```

---

## 🔴 MoE INT8 严重问题

### 1. **`FUSEDMOE_TILEBUF` 硬编码为 256**
**位置**: `w8a8_int8.py:110`

```python
FUSEDMOE_TILEBUF = 256
```

**问题**:
- 这个 magic number 控制临时 buffer 大小
- 如果 `routed_tokens > 256`，会内存越界或性能下降
- 没有注释说明为什么是 256

**示例**:
```python
# DeepSeek-V3: topk=6, batch=128, seq=2048, experts=256
# worst case: 所有 token 都路由到同一个 expert
# routed_tokens_per_expert = 128 * 2048 * 6 / 256 = 6144 >> 256
```

**风险**: 如果 KUTACC kernel 假设 tile <= 256，大 batch 会崩溃或产生错误结果。

**修复建议**:
```python
# 动态计算或从 KUTACC 查询
FUSEDMOE_TILEBUF = min(routed_tokens, kutacc.get_max_tile_size())
# 或至少加注释说明限制
```

---

### 2. **experts_offset 计算未校验单调性**
**位置**: `w8a8_int8.py:150-177`

```python
experts_offset = torch.zeros(num_experts + 1, dtype=torch.int32, device=x.device)
for expert_id in range(num_experts):
    mask = sorted_expert_ids == expert_id
    experts_offset[expert_id + 1] = experts_offset[expert_id] + mask.sum()
```

**问题**:
- 如果 `sorted_expert_ids` 不是真正排序的（bug in sort_tokens_by_expert），offset 会错
- KUTACC kernel 会读错位置 → 数据损坏

**修复建议**:
```python
# 校验单调性
assert torch.all(experts_offset[1:] >= experts_offset[:-1]), \
    "experts_offset must be monotonic"
```

---

### 3. **缺少 topk_weights 归一化检查**
**位置**: `w8a8_int8.py:138-146`

```python
topk_weights = topk_output.topk_weights
if not moe_runner_config.normalize_topk_weights:
    topk_weights = torch.softmax(topk_weights, dim=-1, dtype=torch.float32)
```

**问题**:
- Router 输出的 weights 可能已经是 softmax 后的，或者是 logits
- 如果 `normalize_topk_weights=False` 但 weights 是 logits，会重复 softmax
- 如果 `normalize_topk_weights=True` 但 weights 不是归一化的，combine 时会数值错误

**修复建议**:
```python
# 校验 weights 是否归一化
if moe_runner_config.normalize_topk_weights:
    assert torch.allclose(topk_weights.sum(dim=-1), torch.ones_like(topk_weights[:, 0])), \
        "topk_weights should be normalized when normalize_topk_weights=True"
```

---

## ⚠️ MoE INT8 中等问题

### 4. **量化 scale 重复计算**
**位置**: `w8a8_int8.py:171-184`

```python
x_absmax = x.abs().amax(dim=-1, keepdim=True)
x_scale = (x_absmax / 127.0).clamp(min=1e-12)
x_q = (x / x_scale).round().clamp(-127, 127).to(torch.int8)

# 然后又 pack scale 到 acts_and_scale
torch.ops.async_compute.act_scale_pack_out(x_q, x_scale.view(-1, 1), acts_and_scale)
```

**问题**:
- 每个 token 都计算一次 absmax → scale → quantize
- Down projection 前又重复一次（line 213-219）
- 可以复用 gateup 的 quantization metadata

---

### 5. **routed_scaling_factor 注释不清晰**
**位置**: `w8a8_int8.py:246-248`

```python
# routed_scaling_factor 由外层 DeepseekV2MoE.forward_normal() 统一应用，
# 这里不重复乘，避免 double scaling。
return output.reshape(*orig_shape[:-1], hidden_size)
```

**问题**:
- 这个注释容易让人困惑：`routed_scaling_factor` 到底在哪里应用？
- 如果外层没有应用，这里的输出就是错的
- 应该明确说明调用约定

**建议**:
```python
# NOTE: routed_scaling_factor (通常 0.125) 由调用者 DeepseekV2MoE.forward_normal()
# 在 combine 后统一应用，见 deepseek_v2.py:XXX。这里返回未缩放的输出。
```

---

## 📊 MLA INT8 BMM Review

### 实现

**UK Projection** (Q → compressed Q):
```python
def _batched_gemm_uk(act, weight, weight_scale):
    # act: [M, B, K] BF16 → transpose → [B, M, K] → pack
    # weight: [B, K, N] INT8 → transpose → [B, N, K]
    # out: [B, M, N] BF16
    torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
        packed_act, weight_t, weight_scale, torch.Tensor(), out
    )
```

**UV Projection** (attn_out → hidden):
```python
def _batched_gemm_uv(act, weight, weight_scale):
    # act: [M, B, N] BF16 → transpose → [B, M, N] → pack
    # weight: [B, N, V] INT8 → permute → [B, V, N]
    # out: [B, M, V] BF16
    torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
        packed_act, weight_t, weight_scale, torch.Tensor(), out
    )
```

---

## 🔴 MLA INT8 BMM 严重问题

### 6. **空 tensor 作为 cscale 参数**
**位置**: `w8a8_int8.py:306, 341`

```python
torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
    packed_act, weight_t, weight_scale, torch.Tensor(), out
    #                                    ^^^^^^^^^^^^^^^^^^ 空 tensor
)
```

**问题**:
- `torch.Tensor()` 创建一个空 tensor（numel=0）
- 传给 C++ kernel 作为 `cscale` 参数
- KUTACC 的 `batched_gemm_woqs8` 签名：
  ```cpp
  void batched_gemm_woqs8(..., float *rscale, float *cscale);
  ```
- 如果 kernel 访问 `cscale`，会读到无效指针

**确认需要**:
1. `batched_gemm_woqs8` 是否真的不使用 `cscale`？
2. 如果不使用，应该传 `nullptr`，不是空 tensor

**修复建议**:
```python
# 方案 A: 如果 kernel 不用 cscale，传 None（C++ 侧处理为 nullptr）
torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
    packed_act, weight_t, weight_scale, None, out
)

# 方案 B: 如果 kernel 需要 cscale，创建全 1 的 dummy
cscale = torch.ones((B, M, 1), dtype=torch.float32, device=act.device)
torch.ops.async_compute.batched_gemm_woqs8_allthreads_out(
    packed_act, weight_t, weight_scale, cscale, out
)
```

---

### 7. **注释与代码矛盾**
**位置**: `w8a8_int8.py:282, 319`

**UK 注释**:
```python
# weight: [B, K, N]  INT8, N=kv_lora_rank
```
但代码：
```python
weight_t = weight.transpose(-2, -1).contiguous()  # [B, N, K]
```
说明输入是 `[B, K, N]`，transpose 后变 `[B, N, K]`。但注释里又说 `weight_scale: [B, K, 1]`，这意味着 scale 是对 K 维度的，即 **weight 的行**。

**UV 注释**:
```python
# weight: [B, N, V]  INT8  (post_load_weights 存储的实际布局)
```
但代码：
```python
weight_t = weight.permute(0, 2, 1).contiguous()  # [B, V, N]
```

**问题**: 注释没有区分**逻辑 shape**（语义）和**物理 layout**（内存排布），容易混淆。

**修复建议**:
```python
def _batched_gemm_uk(...):
    """UK projection: Q_nope @ W_kc^T (batched).
    
    Args:
        act: [M, B, K] BF16
             M=seq_len, B=num_heads, K=qk_nope_head_dim
        weight: [B, K, N] INT8 (物理存储 layout)
                逻辑语义: W_kc^T，即 [kv_lora_rank, qk_nope_head_dim]^T per head
        weight_scale: [B, K, 1] FP32 (per row of weight，即 per output channel)
    
    Returns:
        [B, M, N] BF16, 压缩后的 Q
    """
```

---

## ⚠️ MLA INT8 BMM 中等问题

### 8. **缺少 shape 校验**
**位置**: `_batched_gemm_uk` 和 `_batched_gemm_uv`

```python
M, B, K = act.shape
N = weight.shape[-1]
# 没有检查 weight.shape[0] == B
# 没有检查 weight.shape[1] == K (for UK) or N (for UV)
```

**修复建议**:
```python
assert weight.shape[0] == B, f"Batch mismatch: act={B} weight={weight.shape[0]}"
assert weight.shape[1] == K, f"Inner dim mismatch: act={K} weight={weight.shape[1]}"
```

---

## 📊 RMSNorm Review

### 实现

```python
def rmsnorm_forward_kunpeng(self, x, residual=None, post_residual_addition=None):
    if residual is not None:
        if post_residual_addition is not None:
            residual.add_(post_residual_addition)
        torch.ops.async_compute.add_rmsnorm_out(
            x, self.weight, residual, x, self.variance_epsilon
        )
    else:
        torch.ops.async_compute.rmsnorm_out(
            x, self.weight, x, self.variance_epsilon
        )
    return x (or x, residual)
```

---

## ✅ RMSNorm 无明显问题

### 优点
1. ✅ 逻辑清晰：分 with/without residual 两条路径
2. ✅ In-place 优化：复用输入 tensor
3. ✅ 支持 post_residual_addition（DeepSeek 的 shared expert 输出）
4. ✅ 有完善的单元测试（`kunpeng_operators_test.py::test_rmsnorm`）

### 轻微建议

**9. 缺少 epsilon 范围检查**
```python
# 建议在 init 时检查
assert self.variance_epsilon > 0 and self.variance_epsilon < 1e-3, \
    f"variance_epsilon={self.variance_epsilon} 可能不合理"
```

**10. In-place 操作的副作用未文档化**
```python
# x 和 residual 会被原地修改，调用方需要注意
# 建议在 docstring 里说明
```

---

## 🧪 测试覆盖度

### 当前测试 ✅
- ✅ RMSNorm (`kunpeng_operators_test.py`)
- ✅ Router GEMM (`kunpeng_operators_test.py`)
- ✅ Grouped TopK (`kunpeng_topk_test.py`)
- ✅ Argmax (`kunpeng_operators_test.py`)
- ✅ Elementwise (`kunpeng_operators_test.py`)

### 缺失测试 ❌
- ❌ **MoE INT8 gateup/down** - 核心功能无单元测试
- ❌ **MLA INT8 BMM (UK/UV)** - 核心功能无单元测试
- ❌ **MoE token routing** - 排序和 offset 计算逻辑未验证
- ❌ **MoE 边界情况** - topk=1, 单 expert, 空 expert
- ❌ **MLA shape 错误处理** - batch 不匹配是否正确报错

---

## 🎯 修复优先级

### P0 (立即修复)
1. **MLA BMM 空 tensor 参数** - 可能导致 segfault
2. **MoE `FUSEDMOE_TILEBUF` 硬编码** - 大 batch 可能崩溃

### P1 (近期修复)
3. **MoE INT8 测试** - 核心功能未验证
4. **MLA INT8 BMM 测试** - 核心功能未验证
5. **experts_offset 单调性校验** - 防止静默错误
6. **MLA shape 校验** - 防止内存越界

### P2 (可选优化)
7. **注释修正** - 区分逻辑 shape vs 物理 layout
8. **量化 scale 复用** - 减少重复计算
9. **routed_scaling_factor 文档** - 明确调用约定

---

## 📊 总结

### 核心发现

1. **MoE INT8 实现完整**，但有两个高风险问题：
   - `FUSEDMOE_TILEBUF=256` 硬编码
   - 缺少单元测试

2. **MLA INT8 BMM 逻辑正确**，但：
   - 空 tensor 作为 cscale 参数（风险待确认）
   - 注释与代码矛盾，容易误用

3. **RMSNorm 实现优秀**，有测试，无明显问题

### 建议

**立即行动**:
1. 确认 `batched_gemm_woqs8` 是否使用 cscale（查 KUTACC 文档或源码）
2. 补充 MoE INT8 和 MLA INT8 BMM 的单元测试

**近期改进**:
3. 动态计算或文档化 `FUSEDMOE_TILEBUF` 限制
4. 添加 shape 和单调性校验

**当前状态**: 功能实现完整，但**测试覆盖严重不足**，存在潜在的内存安全风险。
