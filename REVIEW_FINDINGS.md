# Kunpeng TP 集成 Code Review 发现

**Review 时间**: 2026-06-13  
**Review 范围**: TP collectives (all_reduce/all_gather)、runtime 生命周期、Python 集成层、测试覆盖度

---

## 🔴 严重问题 (Critical Issues)

### 1. **Runtime 生命周期管理缺失**
**位置**: `kunpeng_communicator.py`、`parallel_state.py`

**问题**: 
- `KunpengCommunicator` 的 `destroy()` 方法**从未被调用**
- `GroupCoordinator` 没有析构逻辑清理 `kunpeng_communicator`
- 每次 server 重启/worker 重新初始化时，`g_runtime` 会泄漏 MPI/KUPL/KUTACC 资源

**影响**:
- MPI communicators 泄漏（`MPI_Comm_free` 未调用）
- KUPL shm windows 泄漏（`kupl_shm_win_free` 未调用）
- 多次 `init_runtime` 会因为 MPI 已初始化但资源未释放而导致不一致状态

**修复建议**:
```python
# parallel_state.py GroupCoordinator 添加
def __del__(self):
    if hasattr(self, 'kunpeng_communicator') and self.kunpeng_communicator:
        self.kunpeng_communicator.destroy()
```

或在 SGLang worker shutdown 路径显式调用 `destroy()`。

---

### 2. **MPI 线程安全未校验**
**位置**: `kupl_runtime.cpp:56`

**问题**:
- `MPI_Init(nullptr, nullptr)` 初始化的是 `MPI_THREAD_SINGLE`
- 但 SGLang 可能是多线程环境（异步 tokenizer、prefill/decode 并发）
- KUTACC 内部用 OpenMP 并行，可能从不同线程访问 MPI

**影响**:
- 如果 SGLang forward 路径是多线程的，MPI 调用可能竞争并导致数据损坏或死锁

**修复建议**:
```cpp
int provided = 0;
MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
TORCH_CHECK(provided >= MPI_THREAD_FUNNELED, 
            "MPI 不支持多线程，当前 level=", provided);
```

或明确文档化「sglang_kupl 要求 single-threaded forward pass」。

---

### 3. **`all_reduce` in-place 路径的数据竞争**
**位置**: `tp_collectives.cpp:139-144`

**问题**:
```cpp
if (input.data_ptr() == output.data_ptr() && is_local_shm_range(...)) {
    // 直接在 shm 上做 all_reduce，无 staging copy
    kutacc::shm_allreduce(..., buffers.data(), ...);
}
```
- 这条路径假设 input tensor **已经在 KUPL shm window 内**
- 但当前 SGLang **从不分配 KUPL shm tensor**，这个分支永远不会执行
- 如果将来实现 shm tensor pool，这个路径的 `make_peer_bf16_buffers` 推导 peer 指针的逻辑**未经测试**

**影响**:
- 当前无影响（dead code）
- 如果启用 shm tensor，可能导致 peer 指针计算错误 → 数据损坏

**修复建议**:
- 删除这个分支（dead code）
- 或添加单元测试验证 shm tensor 路径

---

## ⚠️ 中等问题 (Medium Issues)

### 4. **all_gather 缺少 output shape 预校验**
**位置**: `tp_collectives.cpp:181`

**问题**:
```cpp
TORCH_CHECK(output_size == input_size * runtime.group_size, 
            "tp_all_gather output 最后一维大小不匹配");
```
- 只检查最后一维，**不检查前面维度**
- 如果调用方传错 output shape（比如 batch 维度不一致），KUTACC 会写越界

**示例**:
```python
input = torch.randn(8, 128)   # (batch=8, hidden=128)
output = torch.empty(16, 512) # 错误: batch 应该是 8，不是 16
# 当前检查: 512 == 128*4 ✓ (只检查最后一维)
# 实际: batch 不匹配，KUTACC 会写坏内存
```

**修复建议**:
```cpp
TORCH_CHECK(output.numel() == input.numel() * runtime.group_size,
            "tp_all_gather output numel 不匹配");
TORCH_CHECK(output.size(-1) == input.size(-1) * runtime.group_size,
            "tp_all_gather output 最后一维大小不匹配");
for (int i = 0; i < input.dim() - 1; ++i) {
    TORCH_CHECK(output.size(i) == input.size(i), 
                "tp_all_gather output dim ", i, " 不匹配");
}
```

---

### 5. **对齐检查在 Python 层重复**
**位置**: `tp_collectives.cpp:133` vs `can_use_tp:244`

**问题**:
- `tp_all_reduce` 里已经有对齐检查（line 133）
- `can_use_tp` 里也有对齐检查（line 244）
- 但 **SGLang 从不调用 `can_use_tp`**，直接调 `tp_all_reduce`

**影响**:
- `can_use_tp` 是 dead code
- 维护负担（两处逻辑要同步）

**修复建议**:
- 删除 `can_use_tp` 或在文档里说明它是给「query before call」的 API

---

### 6. **错误信息不友好**
**位置**: `tp_collectives.cpp:134`

**问题**:
```cpp
TORCH_CHECK(..., "tp_all_reduce numel=", input.numel(), " 不满足 KUTACC 的 ", 
            align, " 元素对齐要求 (8 × OMP_NUM_THREADS=", threads, 
            ")。SGLang 将 fallback 到 torch.distributed.all_reduce");
```
- 提到「fallback」，但 **C++ 抛异常后 Python 不会 fallback，直接崩溃**
- 用户看到这个消息会以为 SGLang 自动处理了，实际上 forward 已经挂了

**修复建议**:
```cpp
"tp_all_reduce numel=", input.numel(), " 不满足 KUTACC 的 ", align, 
" 元素对齐要求。请设置 OMP_NUM_THREADS 使 8*threads 能整除 ", input.numel(),
" 或禁用 SGLANG_USE_KUNPENG_TP。"
```

---

### 7. **测试只覆盖 world_size=8**
**位置**: `test/kunpeng_tp_consistency_test.py`

**问题**:
- 测试用 `mpirun -np 8` 硬编码
- **world_size=16 从未测试**（KUTACC allgather 的 16 路模板未验证）
- **不同 OMP_NUM_THREADS 值未测试**（对齐行为依赖 threads）

**修复建议**:
- 参数化测试：`@pytest.mark.parametrize("world_size", [8, 16])`
- 或添加 `mpirun -np 16` 的 CI job

---

## ✅ 轻微问题 (Minor Issues)

### 8. **注释过时**
**位置**: `kunpeng_communicator.py:82`

```python
# DeepSeek R1 INT8 推理只用 dim=-1 (hidden_dim concat)。
# 如果将来需要支持其他 dim，需要在 KUTACC gather 后手动 transpose。
```
- 这段注释应该加到**文档**里，而不是代码注释
- 代码注释应该说明**为什么只支持 dim=-1**（KUTACC 限制），而不是「将来可以扩展」

---

### 9. **日志级别不当**
**位置**: `kunpeng_communicator.py:55`

```python
logger.info("KunpengCommunicator initialized: ...")
```
- 每个 worker 都会打这条 info，8 卡 TP 就是 8 条重复日志
- 应该用 `logger.debug` 或只在 rank 0 打印

---

### 10. **Magic number**
**位置**: `runtime_handle.cpp:5`

```cpp
static constexpr int64_t kRuntimeHandle = 1;
```
- 这个 `1` 的语义是什么？
- 建议重命名 `kSingletonRuntimeHandle` 或加注释「全局单例 runtime 的固定 handle」

---

## 🧪 测试覆盖度分析

### 当前测试覆盖 ✅
- ✅ all_reduce bf16, 多种 shape
- ✅ all_gather bf16, 多种 shape
- ✅ 对齐检查（不满足对齐时报错）
- ✅ rank 顺序一致性（all_gather 数据块顺序）

### 缺失测试 ❌
- ❌ world_size=16
- ❌ all_gather uint8/int8 (代码支持但未测试)
- ❌ world_size=1 (单卡退化路径)
- ❌ 错误输入：shape 不匹配、dtype 错误
- ❌ 边界：numel=0、极大 tensor (超 buffer_size)
- ❌ 资源泄漏：重复 init/destroy
- ❌ 并发：多线程同时调 all_reduce (虽然不应该，但应该测崩溃而不是静默错)

---

## 📋 建议修复优先级

### P0 (立即修复)
1. **Runtime 生命周期管理** — 添加 `__del__` 或显式 `destroy()` 调用
2. **MPI 线程安全** — 使用 `MPI_Init_thread` 或文档化单线程限制

### P1 (近期修复)
3. **all_gather output shape 预校验** — 防止内存越界
4. **删除 in-place all_reduce 的 dead code** — 或添加测试

### P2 (可选优化)
5. **错误信息改进** — 更准确的 fallback 说明
6. **测试扩展** — world_size=16、uint8、边界情况
7. **日志优化** — 降级或去重

---

## 📊 其他算子待 Review

以下算子已实现但未深入 review（需要继续）：
- Dense Linear INT8
- MoE INT8
- MLA INT8 BMM
- RMSNorm
- Grouped TopK
- Router GEMM
- Embedding

---

**Review 结论**: TP collectives 的**核心逻辑正确**，测试通过证明数值一致性和 rank 对齐。主要问题在**资源管理**和**边界检查**，建议优先修复 P0/P1 问题后再部署生产环境。
