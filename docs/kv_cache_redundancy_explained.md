# SGLang KV Cache 在 TP 场景下的存储策略

## 核心答案：是的，每个 rank 都保留完整的 KV cache

---

## 证据链

### 1. **num_kv_heads=1 的含义**

在标准 attention 中：
```python
# 标准 MHA（Multi-Head Attention）
num_kv_heads = num_heads  # 例如 64
cache shape = [seq_len, num_kv_heads, head_dim]
            = [1024, 64, 128]

# TP=4 时，每个 rank 负责部分 heads：
Rank 0: cache [1024, 16, 128]  # heads 0-15
Rank 1: cache [1024, 16, 128]  # heads 16-31
Rank 2: cache [1024, 16, 128]  # heads 32-47
Rank 3: cache [1024, 16, 128]  # heads 48-63
```

在 MLA 中：
```python
# DeepSeek MLA（Multi-head Latent Attention）
num_kv_heads = 1  # ← 关键
cache shape = [seq_len, kv_lora_rank]  # 压缩格式，没有 head 维度
            = [1024, 512]

# TP=4 时，num_kv_heads=1 意味着 cache 不按 head 切分：
Rank 0: cache [1024, 512]  # 完整 cache
Rank 1: cache [1024, 512]  # 完整 cache
Rank 2: cache [1024, 512]  # 完整 cache
Rank 3: cache [1024, 512]  # 完整 cache
```

---

### 2. **代码证据**

**`deepseek_v2.py` line 1580**：
```python
self.attn_mqa = RadixAttention(
    self.num_local_heads,  # Q heads 按 TP 切分：64 / 4 = 16
    self.kv_lora_rank + self.qk_rope_head_dim,
    self.scaling,
    num_kv_heads=1,  # ← KV cache 不切分
    layer_id=layer_id,
    v_head_dim=self.kv_lora_rank,
    ...
)
```

**含义**：
- `num_local_heads = 16`（每个 rank 的 Q heads）
- `num_kv_heads = 1`（**所有 ranks 共享同一个 KV cache 副本**）

---

### 3. **对比标准 attention**

**标准 GQA（Grouped Query Attention）**：
```python
# 例如 Llama-3，num_heads=64, num_kv_heads=8
self.attn = RadixAttention(
    num_heads=16,      # TP=4 时每个 rank 16 个 Q heads
    num_kv_heads=2,    # TP=4 时每个 rank 2 个 KV heads（8/4）
    ...
)

# TP=4 时 cache 切分：
Rank 0: cache [seq_len, 2, head_dim]  # KV heads 0-1
Rank 1: cache [seq_len, 2, head_dim]  # KV heads 2-3
Rank 2: cache [seq_len, 2, head_dim]  # KV heads 4-5
Rank 3: cache [seq_len, 2, head_dim]  # KV heads 6-7
```

**DeepSeek MLA**：
```python
self.attn_mqa = RadixAttention(
    num_heads=16,      # TP=4 时每个 rank 16 个 Q heads
    num_kv_heads=1,    # ← 不切分，每个 rank 都是 1
    ...
)

# TP=4 时 cache **不切分**：
Rank 0: cache [seq_len, kv_lora_rank]  # 完整
Rank 1: cache [seq_len, kv_lora_rank]  # 完整
Rank 2: cache [seq_len, kv_lora_rank]  # 完整
Rank 3: cache [seq_len, kv_lora_rank]  # 完整
```

---

## 为什么 SGLang 这样设计？

### **MLA 的特殊性**

DeepSeek MLA 的 cache 是**压缩的 latent representation**：
```
标准 attention:
  KV_cache = [seq_len, n_heads, head_dim]  (可以按 head 切分)

MLA:
  KV_cache = [seq_len, kv_lora_rank]  (没有 head 维度，无法按 head 切分)
```

### **两种选择**

#### **选择 A：Cache Redundancy（SGLang 当前策略）**
- 每个 rank 存储完整 cache `[seq_len, kv_lora_rank]`
- ✅ 实现简单（不需要 alltoall）
- ❌ 显存冗余（TP=4 时浪费 4x）

#### **选择 B：Cache Parallelism（KunPengDistInfer）**
- Cache 按 **seq 维度切分**：`[seq_len/tp_size, kv_lora_rank]`
- ✅ 显存高效（无冗余）
- ❌ 需要 `mla_q/o_alltoall` 重排 Q/O 矩阵

---

## 实际影响

### **显存占用对比（TP=4，seq_len=1024，kv_lora_rank=512）**

| 策略 | 单个 rank cache 大小 | 总 cache 大小 | 冗余倍数 |
|------|---------------------|--------------|---------|
| **SGLang（冗余）** | 1024 × 512 = 524,288 | 524,288 × 4 = 2,097,152 | **4x** |
| **KunPeng（并行）** | 256 × 512 = 131,072 | 131,072 × 4 = 524,288 | **1x** |

**结论**：SGLang 在 TP=4 时，KV cache 占用是理论最优的 **4 倍**。

---

## 何时成为瓶颈？

### **不成为瓶颈的场景**（当前可接受）

- ✅ **TP=1**（单卡，无冗余）
- ✅ **小 batch**（batch=16，seq_len=2048）
- ✅ **短上下文**（seq_len < 8k）
- ✅ **显存充足**（128GB+ CPU 内存）

### **成为瓶颈的场景**（需要优化）

- ❌ **TP=8**（冗余 8x）
- ❌ **大 batch**（batch=256）
- ❌ **长上下文**（seq_len=128k）
- ❌ **显存紧张**（32GB 内存）

---

## 验证方法

### **方法 1：理论计算**

```python
# DeepSeek-R1 配置
kv_lora_rank = 512
n_layers = 61
seq_len = 2048
batch = 16

# 单个样本的 cache 大小
cache_per_sample = seq_len * kv_lora_rank * n_layers * 2  # 2 bytes (bf16)
                 = 2048 * 512 * 61 * 2
                 = 127 MB

# TP=4 时，每个 rank 的 cache
cache_per_rank = cache_per_sample * batch
               = 127 MB * 16
               = 2 GB

# 总 cache（4 个 ranks）
total_cache = 2 GB * 4 = 8 GB  ← 冗余

# 如果用 cache parallelism（理论最优）
optimal_cache = 2 GB  ← 无冗余
```

### **方法 2：实际测试**

```bash
# 启动 TP=2 推理
export SGLANG_USE_KUNPENG_W8A8=1
torchrun --nproc_per_node=2 -m sglang.launch_server \
  --model-path meituan/DeepSeek-R1-Channel-INT8 \
  --tp 2

# 观察两个 rank 的显存占用
ps aux | grep sglang
# 如果两个进程显存接近（如都是 20GB），说明 cache 冗余
# 如果显存接近减半（如都是 10GB），说明 cache 并行
```

---

## 最终答案

### ✅ **是的，SGLang 的每个 TP rank 都保留完整的 KV cache**

**证据**：
1. `num_kv_heads=1`（不按 head 切分）
2. 没有 `mla_q/o_alltoall` 实现
3. 这是 **cache redundancy** 策略

**影响**：
- TP>1 时显存冗余 N 倍
- 换来实现简单 + 无 alltoall 通信开销
- 大多数场景可接受，极端场景（大 batch + 长上下文 + 高 TP）可能成为瓶颈

**优化方向**：
- 如果遇到显存瓶颈 → 适配 `mla_q/o_alltoall`（切换到 cache parallelism）
- 否则保持当前策略即可
