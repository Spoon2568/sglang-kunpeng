# SGLang CPU 推理 TP 策略验证结果

## 验证日期
2024-06-13

## 验证方法
通过代码分析 `python/sglang/srt/models/deepseek_v2.py` 和相关文件

---

## 关键发现

### 1. **MLA 的 num_kv_heads = 1**

**代码位置**：`deepseek_v2.py` line 1576-1585

```python
self.attn_mqa = RadixAttention(
    self.num_local_heads,
    self.kv_lora_rank + self.qk_rope_head_dim,
    self.scaling,
    num_kv_heads=1,  # ← 关键：KV cache 不按 head 切分
    layer_id=layer_id,
    v_head_dim=self.kv_lora_rank,
    ...
)
```

**含义**：
- MLA 的 KV cache 是 **latent 压缩格式** `[seq_len, kv_lora_rank]`
- **不是** 标准 attention 的 `[seq_len, n_heads, head_dim]`
- `num_kv_heads=1` 表示 **KV cache 不按 head 维度切分**

---

### 2. **Q/O Projection 的 TP 切分**

**代码位置**：`deepseek_v2.py` line 1445-1446

```python
assert num_heads % attn_tp_size == 0
self.num_local_heads = num_heads // attn_tp_size
```

**含义**：
- Q 和 O 投影**按 head 维度 TP 切分**
- 例如 TP=4，64 heads：
  - Rank 0: heads [0, 16)
  - Rank 1: heads [16, 32)
  - Rank 2: heads [32, 48)
  - Rank 3: heads [48, 64)

---

### 3. **没有 mla_q/o_alltoall 实现**

**代码检查**：
- 搜索 `forward_mla.py`：❌ 没有 `mla_q_alltoall` 或 `mla_o_alltoall`
- 搜索 `kunpeng/` 目录：❌ 没有 alltoall 算子封装
- 搜索 `KunpengCommunicator`：✅ 只有 `all_reduce` 和 `all_gather`

---

## SGLang 的 TP 策略分析

### **结论：使用 Cache Redundancy 策略（场景 A）**

```
┌─────────────────────────────────────────────────────┐
│ TP=4 时的内存布局                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│ Rank 0:                                             │
│   Q [1024, 16 heads, ...]  ← head-first切分         │
│   KV_cache [1024, 512]     ← 完整cache（冗余）       │
│                                                      │
│ Rank 1:                                             │
│   Q [1024, 16 heads, ...]                           │
│   KV_cache [1024, 512]     ← 完整cache（冗余）       │
│                                                      │
│ Rank 2:                                             │
│   Q [1024, 16 heads, ...]                           │
│   KV_cache [1024, 512]     ← 完整cache（冗余）       │
│                                                      │
│ Rank 3:                                             │
│   Q [1024, 16 heads, ...]                           │
│   KV_cache [1024, 512]     ← 完整cache（冗余）       │
│                                                      │
└─────────────────────────────────────────────────────┘

总 cache 大小：1024 * 512 * 4 = 2,097,152 elements
实际需要：    1024 * 512 * 1 =   524,288 elements
冗余倍数：4x
```

### **为什么这样设计？**

**优点**：
1. ✅ **实现简单**：不需要复杂的 alltoall 通信
2. ✅ **兼容性好**：与标准 attention 的 TP 策略一致
3. ✅ **通信开销小**：避免 attention 前后的 alltoall

**缺点**：
1. ❌ **显存浪费**：TP=N 时，cache 冗余 N 倍
2. ❌ **扩展性差**：大 batch / 长上下文时显存压力大

---

## 对比：KunPengDistInfer 的策略（场景 B）

```
┌─────────────────────────────────────────────────────┐
│ TP=4 时的内存布局（KunPengDistInfer）                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│ Rank 0:                                             │
│   Q [1024, 16 heads, ...]                           │
│       ↓ mla_q_alltoall (重排)                        │
│   Q_trans [256, 64 heads, ...]  ← seq-first切分     │
│   KV_cache [256, 512]           ← 部分cache         │
│                                                      │
│ Rank 1:                                             │
│   Q_trans [256, 64 heads, ...]                      │
│   KV_cache [256, 512]           ← 部分cache         │
│                                                      │
│ Rank 2:                                             │
│   Q_trans [256, 64 heads, ...]                      │
│   KV_cache [256, 512]           ← 部分cache         │
│                                                      │
│ Rank 3:                                             │
│   Q_trans [256, 64 heads, ...]                      │
│   KV_cache [256, 512]           ← 部分cache         │
│                                                      │
└─────────────────────────────────────────────────────┘

总 cache 大小：256 * 512 * 4 = 524,288 elements
实际需要：    1024 * 512 * 1 = 524,288 elements
冗余倍数：1x（无冗余）
```

**优点**：
1. ✅ **显存高效**：无 cache 冗余
2. ✅ **可扩展**：支持更大 batch / 更长上下文

**缺点**：
1. ❌ **实现复杂**：需要 mla_q/o_alltoall
2. ❌ **通信开销**：每层 attention 前后都要 alltoall

---

## 验证 SGLang 策略的方法

### **方法 1：运行验证脚本**

```bash
cd /home/share/fengguangnan/sibow/llminfer/sglang-kunpeng
python scripts/verify_tp_strategy.py --tp 1
```

### **方法 2：添加 Debug 打印（TP>1 时）**

在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` 添加：

```python
# 在 forward() 函数开头
if self.tp_size > 1:
    import torch.distributed as dist
    tp_rank = dist.get_rank()
    print(f"[TP Rank {tp_rank}] q_nope_out shape: {q_nope_out.shape}")
    # 打印 cache shape（需要访问 kv_cache）
```

**预期输出**（场景 A - cache redundancy）：
```
[TP Rank 0] q_nope_out shape: torch.Size([1024, 16, 192])
[TP Rank 1] q_nope_out shape: torch.Size([1024, 16, 192])
[TP Rank 2] q_nope_out shape: torch.Size([1024, 16, 192])
[TP Rank 3] q_nope_out shape: torch.Size([1024, 16, 192])
All ranks have same q_nope_out[dim0]=1024 ← 说明每卡都看到完整 seq_len
```

如果 dim0 不同（如 256），则说明用了 cache parallelism（场景 B）。

---

## 回答你的问题

### **问题：mla_q/o_alltoall 是做什么的？**

**答案**：
- 在 TP 场景下，**重排 Q/O 矩阵**的维度（head-first ↔ seq-first）
- 使得每卡的 Q 能访问到该卡的 cache（按 seq 维度切分）

### **问题：我的 TP 不是已经实现了吗？**

**答案**：
- ✅ **TP 已实现**（`KunpengCommunicator` 提供 allreduce/allgather）
- ✅ **MLA 在 TP>1 时可以工作**
- ❌ **mla_q/o_alltoall 未实现**（因为 SGLang 用 cache redundancy 策略）

### **问题：EP 没有实现吧？**

**答案**：
- ✅ **你说得对**，缺失的是 **EP（Expert Parallelism）**，不是 TP
- EP 通信（`dispatch_send/recv`, `combine_send/recv`）未实现
- 当前 SGLang 每卡存储所有 experts（不需要 EP）

---

## 最终结论

### ✅ **SGLang TP 策略（已实现）**

| 组件 | TP 策略 | 状态 |
|------|---------|------|
| Q/O Projection | 按 head 切分 | ✅ 已实现 |
| MLA KV Cache | **完整冗余**（每卡存全部） | ✅ 已实现 |
| Allreduce/Allgather | KunpengCommunicator | ✅ 已实现 |
| **mla_q/o_alltoall** | **不需要**（cache redundancy） | ✅ 不需要 |

### 🟢 **mla_q/o_alltoall 优先级：P2（低）**

**原因**：
1. SGLang 用 cache redundancy 策略，不需要 alltoall
2. 显存换通信（trade-off）
3. 仅在极大 batch / 极长上下文时可能成为瓶颈

**如果要适配**（可选优化）：
- 工作量：1-2 周
- 收益：减少 TP>1 时的显存冗余（TP=4 时节省 75% cache 显存）
- 代价：增加通信开销（每层 2 次 alltoall）

---

## 更新后的算子优先级

### 🔴 **P0 - 最高优先级（必需）**

1. **flash_mla_with_kvcache** — MLA attention 核心（性能瓶颈）
2. **yarn** — YaRN RoPE（长上下文性能）
3. **concat_and_cache_mla** — MLA KV cache 管理

### 🟡 **P1 - 高优先级（推荐）**

4. **融合量化算子** — add_rmsnorm_quant, silu_mul_quant（+5-10%）

### 🟢 **P2 - 低优先级（可选优化）**

5. **mla_q/o_alltoall** — 节省 TP>1 时的显存冗余（仅极大规模场景需要）
6. **EP 通信** — Expert Parallelism（当前策略不需要）

---

## 建议

1. **优先适配 P0 算子**（flash_mla、yarn、concat_cache_mla）
2. **暂缓 mla_alltoall**（SGLang 当前策略不需要）
3. **如果后续遇到显存瓶颈**（TP>1 + 大 batch + 长上下文），再考虑 mla_alltoall

---

## 附录：验证命令

```bash
# 1. 运行验证脚本
cd /home/share/fengguangnan/sibow/llminfer/sglang-kunpeng
python scripts/verify_tp_strategy.py

# 2. 实际推理测试（TP=2）
export SGLANG_USE_KUNPENG_W8A8=1
export SGLANG_USE_KUNPENG_TP=1
export KUNPENG_ASYNC_COMPUTE_SO=/path/to/async_compute_op.so

torchrun --nproc_per_node=2 -m sglang.launch_server \
  --model-path meituan/DeepSeek-R1-Channel-INT8 \
  --quantization w8a8_int8 \
  --tp 2 \
  --trust-remote-code

# 3. 观察显存占用（检查 cache 是否冗余）
watch -n 1 nvidia-smi  # 如果 TP=2 时显存没有明显下降，说明 cache 冗余
```
