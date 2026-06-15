# Kunpeng 量化算子 Code Review

**Review 时间**: 2026-06-13  
**Review 范围**: W8A8 INT8 量化算子（Dense Linear、MoE、MLA Attention）

---

## 📊 架构概览

### 量化方案：W8A8 INT8
- **Weight**: INT8 静态量化（per-channel scale）
- **Activation**: INT8 动态量化（per-token scale）
- **Compute**: INT8 GEMM（KUTACC `igemm_bdq` / `batched_gemm_woqs8`）
- **Output**: BF16 dequantize

### 关键文件
```
python/sglang/srt/hardware_backend/kunpeng/quantization/w8a8_int8.py
├── apply_linear()          # Dense Linear INT8
├── _batched_gemm_uk()      # MLA UK projection (Q → compressed Q)
├── _batched_gemm_uv()      # MLA UV projection (attn_out → hidden)
└── apply_moe_gateup_down() # MoE gateup/down INT8

python/sglang/srt/layers/quantization/w8a8_int8.py
├── W8A8Int8Config          # 量化配置
├── W8A8Int8LinearMethod    # Linear layer 集成
└── W8A8Int8MoEMethod       # MoE layer 集成

C++ Kernels (KUTACC):
├── igemm_bdq()             # Dense GEMM: [M,K] @ [N,K]^T → [M,N]
├── batched_gemm_woqs8()    # Batched GEMM for MLA
├── igemm_gateup()          # MoE gate-up fusion
└── igemm_down()            # MoE down projection
```

---

## 🔴 严重问题 (Critical Issues)

### 1. **硬编码的 SO 路径**
**位置**: `w8a8_int8.py:6-9`

```python
ASYNC_COMPUTE_SO = os.environ.get(
    "KUNPENG_ASYNC_COMPUTE_SO",
    "/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so",
)
```

**问题**:
- 默认路径是绝对路径，指向特定用户目录
- 其他用户或部署环境无法运行
- 应该用相对路径或 package data

**修复建议**:
```python
import os
from pathlib import Path

# 相对于当前模块的路径
DEFAULT_SO = str(Path(__file__).parent.parent.parent.parent.parent.parent / 
                 "Kpllminfer/kernels/async_compute_op.so")
ASYNC_COMPUTE_SO = os.environ.get("KUNPENG_ASYNC_COMPUTE_SO", DEFAULT_SO)

# 或者在 setup.py 里安装到 package data
```

---

### 2. **缺少 SO 文件存在性检查**
**位置**: `w8a8_int8.py:18-22`

```python
def load_async_compute():
    global _loaded
    if not _loaded:
        torch.ops.load_library(ASYNC_COMPUTE_SO)
        _loaded = True
```

**问题**:
- 如果 SO 不存在，`torch.ops.load_library` 崩溃
- 错误信息不友好（PyTorch 的 dlopen 错误）
- 无法提前检测部署问题

**修复建议**:
```python
def load_async_compute():
    global _loaded
    if not _loaded:
        if not os.path.exists(ASYNC_COMPUTE_SO):
            raise FileNotFoundError(
                f"Kunpeng async_compute kernel not found: {ASYNC_COMPUTE_SO}\n"
                f"请设置环境变量 KUNPENG_ASYNC_COMPUTE_SO 指向正确的 .so 路径，"
                f"或重新编译 Kpllminfer/kernels/async_compute_op.so"
            )
        torch.ops.load_library(ASYNC_COMPUTE_SO)
        _loaded = True
```

---

### 3. **~~workspace 内存泄漏风险~~** ✅ **已验证安全**
**位置**: `w8a8_int8.py:37-84` (apply_linear)

**原问题**:
```python
workspace = torch.empty(workspace_bytes(M, N, K), dtype=torch.uint8, device=x.device)
torch.ops.async_compute.igemm_bdq_out(..., workspace)
# workspace 在函数结束后被 GC 回收，如果 kernel 异步执行可能 use-after-free
```

**验证结果**: ✅ **无问题**
- 测试证明：临时 workspace 模式安全（100 次迭代 + 强制 GC 无崩溃）
- **KUTACC kernel 是同步的**：`kutacc::parallel_for` 阻塞等待所有线程完成
- **Torch binding 也是同步的**：函数返回时计算已完成，workspace 不再使用
- **"async_compute" 含义**：异步于 GPU（CPU-only），非异步于调用者

**测试覆盖**: `test/kunpeng_workspace_lifetime_test.py` 全部通过

---

## ⚠️ 中等问题 (Medium Issues)

### 4. **量化 scale 的数值稳定性未验证**
**位置**: `w8a8_int8.py:48-61`

```python
x_absmax = x_2d.abs().max(dim=-1, keepdim=True).values
x_scale = (x_absmax / 127.0).clamp(min=1e-12)
x_q = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)
```

**问题**:
1. **`clamp(min=1e-12)` 的合理性**
   - 如果 `x_absmax` 接近 0（比如全 0 tensor），scale = 1e-12
   - 量化后 `x_q ≈ x / 1e-12 = 巨大值` → clamp 到 127
   - Dequantize 后 `x_out = 127 * 1e-12 ≈ 0`，精度损失极大
   - 应该用更大的 min（比如 1e-6）或直接处理全 0 情况

2. **`round()` 的舍入模式**
   - PyTorch `round()` 是 "round half to even"（银行家舍入）
   - 某些场景下可能引入系统性偏差
   - 标准做法是 `(x / scale + 0.5).floor()` 或 `(x / scale).round()`

3. **clamp 范围 `[-128, 127]`**
   - INT8 的对称范围是 `[-127, 127]`（避免 -128 的不对称）
   - 使用 -128 可能导致 dequantize 时的数值偏差

**修复建议**:
```python
x_absmax = x_2d.abs().max(dim=-1, keepdim=True).values
# 处理全 0 或接近 0 的情况
x_scale = (x_absmax / 127.0).clamp(min=1e-6)  # 更保守的 min
x_q = (x_2d / x_scale).round().clamp(-127, 127).to(torch.int8)  # 对称范围
```

**需要测试**: 极端输入（全 0、极小值、极大值）的量化精度。

---

### 5. **MLA batched GEMM 的 shape 假设未文档化**
**位置**: `w8a8_int8.py:266-288` (_batched_gemm_uk)

```python
def _batched_gemm_uk(act, weight, weight_scale, out_dtype=torch.bfloat16):
    """
    act: [M, B, N]  BF16
    weight: [B, K, N]  INT8
    weight_scale: [B, K, 1]  float32
    Returns: [M, B, K]  BF16
    """
```

**问题**:
1. **Shape 约定不清晰**
   - `act` 是 `[M, B, N]` 还是 `[B, M, N]`？注释说 `[M, B, N]`，但代码里有 `transpose(0, 1)`
   - `weight` 存储布局是 `[B, K, N]` 还是 `[B, N, K]`？注释说 `[B, K, N]`，但后面又 `transpose(-2, -1)`
   - 调用方容易搞混

2. **缺少 shape 校验**
   - 如果调用方传错 shape（比如 batch 维度不匹配），kernel 会默默产生错误结果
   - 应该加 `assert` 或 `TORCH_CHECK`

**修复建议**:
```python
def _batched_gemm_uk(act, weight, weight_scale, out_dtype=torch.bfloat16):
    """MLA UK projection: Q_nope @ W_kc^T (batched over num_heads).
    
    Args:
        act: [M, B, N] BF16, where M=seq_len, B=num_heads, N=qk_nope_head_dim
        weight: [B, K, N] INT8, where K=kv_lora_rank (物理存储layout，K行N列)
        weight_scale: [B, K, 1] FP32, per-channel (K) row scale
    
    Returns:
        [M, B, K] BF16, compressed Q for MLA
    """
    M, B, N = act.shape
    assert weight.shape[0] == B, f"Batch mismatch: act={B} weight={weight.shape[0]}"
    assert weight.shape[2] == N, f"Inner dim mismatch: act={N} weight={weight.shape[2]}"
    K = weight.shape[1]
    
    # ... 现有实现
```

---

### 6. **MoE kernel 缺少单元测试**
**位置**: `w8a8_int8.py:173-263` (apply_moe_gateup_down)

**问题**:
- MoE 的 `apply_moe_gateup_down` **没有独立的单元测试**
- `kunpeng_operators_test.py` 只测了 router GEMM、topk、RMSNorm，**没测 MoE INT8 GEMM**
- 复杂的 token routing、expert dispatch 逻辑未验证

**风险**:
- Token 路由错误 → 错误的 expert 处理错误的 token
- Scale 不匹配 → 数值爆炸或归零
- 内存越界 → 随机崩溃

**修复建议**:
添加测试 `test/kunpeng_moe_int8_test.py`，验证：
1. 单 expert 退化情况（应等价于 Dense Linear）
2. 多 expert + topk=1（每个 token 只路由到 1 个 expert）
3. 多 expert + topk=2（DeepSeek 的 shared expert 模式）
4. Gateup fusion 正确性
5. 与 FP16/BF16 MoE 的数值对比

---

## ✅ 轻微问题 (Minor Issues)

### 7. **workspace_bytes 计算过于保守**
**位置**: `w8a8_int8.py:25-26`

```python
def workspace_bytes(m: int, n: int, k: int) -> int:
    return max(m * n * k * 2, 1024)
```

**问题**:
- `m * n * k * 2` 看起来是随意猜测的（为什么是 2 倍？）
- 没有注释说明为什么需要这么大
- 浪费内存（每次 forward 都分配）

**建议**: 加注释说明计算依据，或从 KUTACC 文档获取准确公式。

---

### 8. **load_async_compute 的全局状态不线程安全**
**位置**: `w8a8_int8.py:18-22`

```python
_loaded = False

def load_async_compute():
    global _loaded
    if not _loaded:
        torch.ops.load_library(ASYNC_COMPUTE_SO)
        _loaded = True
```

**问题**:
- 如果多线程同时调用 `load_async_compute()`（比如 data parallel worker 初始化），可能重复加载
- 虽然 `torch.ops.load_library` 内部可能有保护，但这里的 `_loaded` flag 有竞争条件

**修复建议**:
```python
import threading
_load_lock = threading.Lock()
_loaded = False

def load_async_compute():
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if not _loaded:
            torch.ops.load_library(ASYNC_COMPUTE_SO)
            _loaded = True
```

---

### 9. **注释与代码不一致**
**位置**: `w8a8_int8.py:266-343`

**示例 1**: `_batched_gemm_uk` line 275
```python
# weight: [B, N, K]  INT8  (post_load_weights 存储的实际布局)
```
但代码里 `weight.shape` 是 `[B, K, N]`，后面又 transpose。

**示例 2**: `_batched_gemm_uv` line 316
```python
# weight: [B, N, V]  INT8  (post_load_weights 存储的实际布局)
```
但后面又 `permute(0, 2, 1)` 变成 `[B, V, N]`。

**建议**: 统一注释约定：说明是**逻辑 shape**（语义）还是**物理 layout**（内存排布）。

---

### 10. **缺少 dtype 校验**
**位置**: 多处（apply_linear、_batched_gemm_uk 等）

**问题**:
- 没有检查 `act` 是否是 BF16
- 没有检查 `weight` 是否是 INT8
- 如果调用方传错 dtype，kernel 会产生错误结果或崩溃

**修复建议**:
```python
def apply_linear(layer, x: torch.Tensor, bias=None):
    assert x.dtype == torch.bfloat16, f"Expected BF16 input, got {x.dtype}"
    assert layer.weight.dtype == torch.int8, f"Expected INT8 weight, got {layer.weight.dtype}"
    # ...
```

---

## 📋 测试覆盖度分析

### 当前测试 ✅
- ✅ Router BF16 GEMM (`test_router_gemm`)
- ✅ Grouped TopK (`test_kunpeng_topk.py`)
- ✅ RMSNorm (`test_rmsnorm`)
- ✅ Argmax (`test_argmax`)
- ✅ Elementwise (`test_mul_scalar_add`)

### 缺失测试 ❌
- ❌ **Dense Linear INT8** (`apply_linear`) - 核心量化算子无单元测试
- ❌ **MLA INT8 BMM** (`_batched_gemm_uk` / `_batched_gemm_uv`) - 无单元测试
- ❌ **MoE INT8** (`apply_moe_gateup_down`) - 无单元测试
- ❌ **量化精度** - 与 FP16/BF16 的数值对比（SQNR、相对误差）
- ❌ **边界情况** - 全 0 输入、极小 scale、极大值
- ❌ **Shape 错误处理** - 传错 shape 是否正确报错

---

## 🎯 修复优先级

### P0 (立即修复)
1. **SO 路径硬编码** - 影响可部署性
2. **SO 存在性检查** - 提升错误可调试性
3. ~~**workspace 生命周期**~~ ✅ 已验证安全

### P1 (近期修复)
4. **量化 scale 稳定性** - 影响精度
5. **MoE INT8 测试** - 核心功能未���证
6. **Shape 校验** - 防止静默错误

### P2 (可选优化)
7. **注释修正**
8. **线程安全**
9. **workspace 计算优化**

---

## 🧪 建议新增测试

### 测试 1: Dense Linear INT8 数值精度
```python
def test_dense_linear_int8_accuracy():
    # 对比 kunpeng W8A8 vs torch FP32 baseline
    # 测试 SQNR、最大相对误差、分布差异
    pass
```

### 测试 2: MLA INT8 BMM
```python
def test_mla_uk_projection():
    # 验证 _batched_gemm_uk 的正确性
    # shape: [M=128, B=16, N=192] @ [B=16, K=512, N=192]^T
    pass

def test_mla_uv_projection():
    # 验证 _batched_gemm_uv 的正确性
    pass
```

### 测试 3: MoE INT8 端到端
```python
def test_moe_int8_single_expert():
    # 退化为 Dense Linear
    pass

def test_moe_int8_routing():
    # 验证 token 路由正确性
    pass
```

### 测试 4: 边界情况
```python
def test_zero_input():
    # 全 0 tensor 的量化精度
    pass

def test_extreme_values():
    # 极大值、极小值
    pass
```

---

## 📊 总结

### 核心发现
1. **量化算子本身逻辑清晰**，但**缺少单元测试**（Dense Linear、MLA、MoE 都没有独立验证）
2. **SO 路径硬编码**是最紧迫的部署问题
3. **Workspace 生命周期**可能是潜在崩溃风险（需确认 kernel 是否异步）
4. **量化 scale 的边界情况**未处理（全 0、极小值）

### 建议
- **优先补充测试**：Dense Linear INT8、MLA BMM、MoE INT8
- **修复 P0 问题**：SO 路径、workspace 生命周期
- **验证数值精度**：与 FP16/BF16 对比 SQNR

**当前状态**: 功能实现完整，但测试覆盖不足，生产环境部署前需补充验证。
