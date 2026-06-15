# YaRN RoPE 算子适配总结

## ✅ 完成状态

**适配日期**：2024-06-13  
**测试结果**：6/6 测试全部通过 ✅

---

## 📦 实现内容

### 1. **kunpeng/rope.py** — 纯 kunpeng 实现

#### `yarn_init_cache_forward()`
```python
def yarn_init_cache_forward(
    dim: int,
    max_position_embeddings: int,
    base: float,
    scaling_factor: float,
    beta_fast: int = 32,
    beta_slow: int = 1,
    extrapolation_factor: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    attn_factor: float = 1.0,
) -> torch.Tensor:
    """Initialize YaRN cos/sin cache.
    
    Returns: [max_position_embeddings, dim] bf16 tensor
    Format: [cos[0:dim//2], sin[0:dim//2]] (first half cos, second half sin)
    """
```

**作用**：
- 预计算所有位置的 YaRN-scaled cos/sin 值
- 仅在模型初始化时调用一次
- 支持 DeepSeek 的所有 YaRN 参数

**调用**：`torch.ops.async_compute.yarn_init_cache_out`

---

#### `yarn_forward()`
```python
def yarn_forward(
    q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    position_ids: torch.Tensor,  # [batch, seq_len] int64
    cos_sin_cache: torch.Tensor,  # [max_position, dim] bf16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply YaRN RoPE to Q and K tensors."""
```

**作用**：
- 在每次前向传播时应用 RoPE 旋转
- 基于 position_ids 从 cache 查表
- 支持任意 batch size 和 sequence length

**调用**：`torch.ops.async_compute.yarn_out`

---

### 2. **test/kunpeng_yarn_test.py** — 完备测试

#### Cache 初始化测试（3 个）
- ✅ `test_small_cache` — 小配置（dim=64, max_pos=128）
- ✅ `test_deepseek_r1_config` — DeepSeek-R1 配置
- ✅ `test_large_dim` — 大 head_dim（dim=128）

#### RoPE 应用测试（3 个）
- ✅ `test_single_token` — 单 token
- ✅ `test_batch_sequence` — batch + 多 tokens（batch=4, seq=16）
- ✅ `test_long_context` — 长上下文（position=8192）

**验证方法**：与 PyTorch 参考实现对比（bf16 tolerance: atol=0.02, rtol=0.02）

---

## 🔍 关键技术细节

### 1. **Cache 格式**

C++ 格式（非交错）：
```
[cos[0], cos[1], ..., cos[dim/2-1], sin[0], sin[1], ..., sin[dim/2-1]]
 ← 前半部分：cos          ← 后半部分：sin
```

**不是**交错格式：~~`[cos[0], sin[0], cos[1], sin[1], ...]`~~

---

### 2. **YaRN 算法实现**

C++ 的 YaRN 实现（line 234-244 in yarn_init_cache.cc）：

```cpp
double interpolation = pid / scaling_factor;
double extrapolation = pid;
for (int i = 0; i < dim / 2; ++i) {
    double mask = (1.0 - clamp((i - low) / (high - low), 0.0, 1.0)) * extrapolation_factor;
    double theta = (1.0 - mask) * interpolation + mask * extrapolation;
    c[i] = cos(theta) * real_mscale;
    s[i] = sin(theta) * real_mscale;
    extrapolation *= theta_scale;  // theta_scale = base^(-2/dim)
    interpolation *= theta_scale;
}
```

**关键点**：
1. **theta_scale** = `base^(-2/dim)`（不是标准 RoPE 的 `1/base^(2i/dim)`）
2. **Interpolation/Extrapolation 混合**：根据频率索引 `i` 和 YaRN mask 决定
3. **mscale**：magnitude scaling，防止高频信息丢失

---

### 3. **与标准 RoPE 的区别**

| 维度 | 标准 RoPE | YaRN RoPE |
|------|----------|-----------|
| 频率计算 | `freq_i = 1 / base^(2i/dim)` | 分频段混合 interpolation/extrapolation |
| 长上下文 | 直接外推（性能下降） | YaRN 缩放（性能保持） |
| 参数 | `base`, `dim` | `base`, `dim`, `scaling_factor`, `beta_fast/slow`, `mscale` |
| DeepSeek-R1 | — | 支持 160k 上下文 |

---

## 🎯 性能预期

### **长上下文场景**

YaRN 的优势在长上下文：

| 场景 | 标准 RoPE | YaRN RoPE |
|------|----------|-----------|
| 2k tokens | 性能相当 | 性能相当 |
| 8k tokens | 开始下降 | 保持稳定 |
| 32k tokens | 显著下降 | 保持稳定 |
| 128k tokens | 不可用 | 可用 ✅ |

**DeepSeek-R1** 通过 YaRN 支持 **160k 上下文**（`scaling_factor=40.0`）。

---

## 📊 测试结果

```bash
export KUNPENG_ASYNC_COMPUTE_SO=/path/to/async_compute_op.so
export SGLANG_USE_KUNPENG_W8A8=1
python test/kunpeng_yarn_test.py

test_batch_sequence ... ok
test_long_context ... ok
test_single_token ... ok
test_deepseek_r1_config ... ok
test_large_dim ... ok
test_small_cache ... ok

Ran 6 tests in 5.929s

OK ✅
```

---

## 🔧 集成到 SGLang（下一步）

### **当前 SGLang 使用**

`python/sglang/srt/layers/rotary_embedding/rope_variant.py` line 335-395：
- `YarnRotaryEmbedding` 类（PyTorch 实现）
- 在 `forward_mla.py` 中调用

### **替换方案**

#### 方案 A：直接替换（推荐）
```python
# In forward_mla.py
from sglang.srt.hardware_backend.kunpeng.quantization.w8a8_int8 import use_kunpeng_w8a8

if use_kunpeng_w8a8():
    from sglang.srt.hardware_backend.kunpeng.rope import yarn_forward
    q_pe_out, k_pe_out = yarn_forward(q_pe, k_pe, position_ids, self.yarn_cache)
else:
    # 原 PyTorch 路径
    q_pe_out, k_pe_out = self.rotary_emb(q_pe, k_pe, position_ids)
```

#### 方案 B：修改 YarnRotaryEmbedding 类
在 `YarnRotaryEmbedding.__init__` 和 `forward` 中加 kunpeng 分支。

---

## 📝 待办事项

### ✅ 已完成
1. 封装 `yarn_init_cache_forward` 和 `yarn_forward`
2. 编写 6 个测试（全部通过）
3. 验证与 C++ 实现的一致性

### ⏳ 下一步
1. 集成到 `forward_mla.py`（替换 PyTorch YaRN）
2. 端到端测试（DeepSeek-R1 推理）
3. 性能测试（长上下文场景）

### 📌 注意事项
- YaRN cache 在模型初始化时预计算，需要确定何时调用 `yarn_init_cache_forward`
- `forward_mla.py` 可能有多个 RoPE 调用点，需要逐一替换
- 验证 TP > 1 时的正确性（RoPE 不涉及跨卡通信，应该天然支持）

---

## 🎉 总结

### ✅ **纯 kunpeng 实现成功**

- 无 PyTorch fallback
- 完全依赖 `torch.ops.async_compute.yarn_out` 和 `yarn_init_cache_out`
- 6/6 测试通过

### 🚀 **预期收益**

- RoPE 计算加速（kunpeng SVE 优化）
- 长上下文性能保持（YaRN 算法）
- 为后续 MLA attention 适配打下基础

### 📚 **学到的经验**

1. **Cache 格式重要**：C++ 用非交错格式，测试需匹配
2. **YaRN 算法复杂**：需要完全理解 C++ 实现逻辑才能写正确的参考
3. **纯 kunpeng 可行**：无需 fallback，简化代码

---

## 📖 参考资料

- YaRN 论文：https://arxiv.org/abs/2309.00071
- DeepSeek-V2 论文：Section on YaRN RoPE
- C++ 实现：`Kpllminfer/async_rt/csrc/compute_ops/rope/yarn_init_cache.cc`
