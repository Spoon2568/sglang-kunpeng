# Kunpeng 测试套件文档

本文档说明如何运行 Kunpeng SGLang 集成的所有测试。

---

## 📋 测试清单

### ✅ 已实现的测试

#### 单机测试（不需要 MPI）

| 测试文件 | 测试内容 | 状态 |
|---------|---------|------|
| `kunpeng_workspace_lifetime_test.py` | Workspace 生命周期安全性 | ✅ |
| `kunpeng_moe_tilebuf_test.py` | MoE TILEBUF 大小验证 | ✅ |
| `kunpeng_mla_bmm_cscale_test.py` | MLA BMM 空 cscale 参数验证 | ✅ |
| `kunpeng_operators_test.py` | RMSNorm, Router GEMM, TopK, Argmax 等 | ✅ |
| `kunpeng_topk_test.py` | Grouped TopK 正确性 | ✅ |
| `kunpeng_moe_int8_test.py` | **MoE INT8 单元测试** | ✅ **新增** |
| `kunpeng_mla_int8_bmm_test.py` | **MLA INT8 BMM 单元测试** | ✅ **新增** |

#### 分布式测试（需要 MPI）

| 测试文件 | NP | 测试内容 | 状态 |
|---------|---|---------|------|
| `kunpeng_lifecycle_test.py` | 4 | Runtime 生命周期管理 | ✅ |
| `kunpeng_tp_consistency_test.py` | 8 | TP all_reduce/all_gather 一致性 | ✅ |
| `kunpeng_allgather_shape_test.py` | 8 | all_gather shape 校验 | ✅ |

---

## 🚀 快速开始

### 运行所有测试（推荐）

```bash
cd /home/share/fengguangnan/sibow/llminfer/sglang-kunpeng
source .venv/bin/activate
bash scripts/run_kunpeng_tests.sh
```

### 运行单个测试

#### 单机测试
```bash
# 激活环境
source .venv/bin/activate
export KUNPENG_ASYNC_COMPUTE_SO=/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so

# 运行测试
python test/kunpeng_moe_int8_test.py
python test/kunpeng_mla_int8_bmm_test.py
```

#### 分布式测试
```bash
# 激活环境
source .venv/bin/activate
export SGLANG_USE_KUNPENG_TP=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 运行测试
mpirun -np 4 python test/kunpeng_lifecycle_test.py
mpirun -np 8 python test/kunpeng_tp_consistency_test.py
```

---

## 📊 测试覆盖度

### TP Collectives ✅ 优秀

| 模块 | 功能测试 | 数值验证 | 边界测试 | 状态 |
|------|---------|---------|---------|------|
| all_reduce | ✅ | ✅ | ✅ | 完整 |
| all_gather | ✅ | ✅ | ✅ | 完整 |
| Runtime 生命周期 | ✅ | N/A | ✅ | 完整 |
| Workspace 安全性 | ✅ | N/A | ✅ | 完整 |

### Dense Linear INT8 ⚠️ 良好

| 模块 | 功能测试 | 数值验证 | 边界测试 | 状态 |
|------|---------|---------|---------|------|
| apply_linear | ✅ | ⚠️ 缺失 | ⚠️ 缺失 | 部分 |
| Workspace 生命周期 | ✅ | N/A | ✅ | 完整 |

**建议**: 补充 Dense Linear INT8 的精度测试（vs FP32 baseline）

### MoE INT8 ✅ 优秀（新增）

| 模块 | 功能测试 | 数值验证 | 边界测试 | 状态 |
|------|---------|---------|---------|------|
| Gateup/down | ✅ | ✅ | ✅ | 完整 |
| Token routing | ✅ | ✅ | ✅ | 完整 |
| Load imbalance | ✅ | N/A | ✅ | 完整 |
| TILEBUF 验证 | ✅ | N/A | ✅ | 完整 |

**测试用例**:
- ✅ 单 expert + topk=1（退化为 Dense Linear）
- ✅ 多 expert + topk=2（典型配置）
- ✅ 极端 load imbalance（所有 token 到一个 expert）
- ✅ experts_offset 单调性校验

### MLA INT8 BMM ✅ 优秀（新增）

| 模块 | 功能测试 | 数值验证 | 边界测试 | 状态 |
|------|---------|---------|---------|------|
| UK projection | ✅ | ✅ | ✅ | 完整 |
| UV projection | ✅ | ✅ | ✅ | 完整 |
| Shape 变换 | ✅ | N/A | ✅ | 完整 |
| cscale 参数 | ✅ | N/A | ✅ | 完整 |

**测试用例**:
- ✅ UK projection 正确性（vs BF16 baseline）
- ✅ UV projection 正确性（vs BF16 baseline）
- ✅ 小 shape 边界情况
- ✅ 大 shape（DeepSeek-V3 规模）
- ✅ 量化精度分析（SQNR、相对误差）

### RMSNorm ✅ 优秀

| 模块 | 功能测试 | 数值验证 | 边界测试 | 状态 |
|------|---------|---------|---------|------|
| rmsnorm_forward | ✅ | ✅ | ✅ | 完整 |
| Residual fusion | ✅ | ✅ | ✅ | 完整 |

---

## 🎯 测试结果预期

### 正常输出示例

```
====================================================================
Kunpeng SGLang 测试套件
====================================================================

✓ 虚拟环境已激活
✓ SO 文件: /path/to/async_compute_op.so

====================================================================
单机测试（不需要 MPI）
====================================================================

----------------------------------------------------------------------
运行: test/kunpeng_moe_int8_test.py
----------------------------------------------------------------------
test_single_expert_topk1 (__main__.TestMoEINT8)
Test: single expert, topk=1 (should be equivalent to Dense Linear). ... ok
test_multi_expert_topk2 (__main__.TestMoEINT8)
Test: 4 experts, topk=2 (typical DeepSeek config). ... ok
test_load_imbalance (__main__.TestMoEINT8)
Test: extreme load imbalance (all tokens to expert 0). ... ok
test_experts_offset_monotonic (__main__.TestMoEINT8)
Test: experts_offset is monotonic increasing. ... ok

Ran 4 tests in 2.345s

OK
✓ PASS: test/kunpeng_moe_int8_test.py

====================================================================
测试总结
====================================================================

单机测试: 7 passed, 0 failed
分布式测试: 3 passed, 0 failed

总计: 10/10 passed

====================================================================
✓ 所有测试通过！
====================================================================
```

---

## 🧪 测试详细说明

### MoE INT8 测试 (`kunpeng_moe_int8_test.py`)

**测试目标**: 验证 MoE INT8 量化的正确性和健壮性

#### Test 1: 单 expert + topk=1
- **目的**: 验证退化为 Dense Linear 时的正确性
- **方法**: 与手动 BF16 计算对比
- **阈值**: max_abs < 0.2, max_rel < 0.1

#### Test 2: 多 expert + topk=2
- **目的**: 验证典型 DeepSeek 配置（4 experts, topk=2）
- **方法**: 检查输出合理性（无 NaN/Inf）
- **验证**: routed_tokens = num_tokens * topk

#### Test 3: Load imbalance
- **目的**: 验证极端情况（256 tokens 到单个 expert，4x TILEBUF）
- **方法**: 检查 TILEBUF=64 是否足够
- **验证**: 无崩溃、无数值异常

#### Test 4: experts_offset 单调性
- **目的**: 验证 token 排序和 offset 计算逻辑
- **方法**: 检查 `experts_offset[i+1] >= experts_offset[i]`
- **验证**: 总 routed_tokens 正确

---

### MLA INT8 BMM 测试 (`kunpeng_mla_int8_bmm_test.py`)

**测试目标**: 验证 MLA INT8 BMM 的数值精度和 shape 正确性

#### Test 1-2: UK/UV projection 正确性
- **目的**: 与 BF16 baseline 对比
- **方法**: 手动计算 BF16 结果，对比 INT8 输出
- **阈值**: max_abs < 0.15, max_rel < 0.08

#### Test 3-4: 小/大 shape 边界测试
- **目的**: 验证各种 shape 都能正确处理
- **小 shape**: M=16, B=4 （最小合理配置）
- **大 shape**: M=512, B=32 （DeepSeek-V3 规模）

#### Test 5-6: 量化精度分析
- **目的**: 量化精度分析（SQNR、相对误差）
- **指标**:
  - Mean absolute error < 0.05
  - Mean relative error < 0.03
  - SQNR > 30 dB

#### Test 7-8: Shape 校验
- **目的**: 验证正确 shape 能通过
- **方法**: 测试典型 DeepSeek-V3 shape

---

## 📈 持续集成建议

### CI 流程

```yaml
# .github/workflows/kunpeng-tests.yml (示例)
name: Kunpeng Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: kunpeng-runner
    steps:
      - uses: actions/checkout@v2
      - name: Setup environment
        run: |
          source .venv/bin/activate
      - name: Run single-node tests
        run: |
          bash scripts/run_kunpeng_tests.sh
```

### 回归测试

建议在以下情况运行完整测试套件：
1. 修改 TP collectives 代码
2. 修改量化算子代码
3. 升级 KUTACC 版本
4. 修改 SGLang 核心逻辑

---

## 🐛 故障排查

### 常见问题

#### 问题 1: `FileNotFoundError: async_compute_op.so not found`
**解决**: 设置环境变量
```bash
export KUNPENG_ASYNC_COMPUTE_SO=/path/to/async_compute_op.so
```

#### 问题 2: MPI 测试失败 `mpirun: command not found`
**解决**: 安装 MPI 或跳过分布式测试
```bash
# 只运行单机测试
python test/kunpeng_moe_int8_test.py
python test/kunpeng_mla_int8_bmm_test.py
```

#### 问题 3: 数值精度测试偶尔失败
**原因**: 随机初始化导致的数值波动
**解决**: 设置随机种子（测试已包含 `torch.manual_seed(42)`）

#### 问题 4: `MASTER_PORT already in use`
**解决**: 换一个端口
```bash
export MASTER_PORT=29501
mpirun -np 8 python test/kunpeng_tp_consistency_test.py
```

---

## 📚 参考文档

- [FINAL_REVIEW_SUMMARY.md](FINAL_REVIEW_SUMMARY.md) - 完整 code review 报告
- [QUANTIZATION_REVIEW.md](QUANTIZATION_REVIEW.md) - 量化算子详细 review
- [OTHER_OPERATORS_REVIEW.md](OTHER_OPERATORS_REVIEW.md) - MoE/MLA/RMSNorm review

---

## ✅ 测试验收标准

部署到生产环境前，需要满足：

### 必须通过（Blocking）
- ✅ 所有 TP collectives 测试通过
- ✅ Runtime 生命周期测试通过
- ✅ Workspace 安全性测试通过
- ✅ MoE INT8 单元测试通过
- ✅ MLA INT8 BMM 单元测试通过
- ✅ RMSNorm 测试通过

### 建议通过（Non-blocking）
- ⚠️ Dense Linear INT8 精度测试（当前缺失）
- ⚠️ 端到端推理验证（需要完整模型）

---

**测试套件维护者**: Claude Opus 4.7  
**最后更新**: 2026-06-13  
**测试总数**: 10 个文件，50+ 测试用例
