# DeepSeek-R1 Channel-INT8 自定义 C++ 后端适配清单

本文针对 `meituan/DeepSeek-R1-Channel-INT8` 在未适配硬件上的 SGLang 推理适配。目标是保留现有 DeepSeek 模型结构和调度逻辑，只替换底层 INT8 算子、MoE 算子、MLA BMM 和分布式通信。

核心思路：

- 不重写 DeepSeek-R1 模型。
- C++ 算子统一注册到 `sgl-kernel`。
- Dense linear 和 MoE 通过 quantization method 接入。
- TP/EP 通信通过 `GroupCoordinator` 和 communicator 接入。
- DeepSeek 专用改动只放在 MLA 权重后处理和 MLA forward 路径中。

## 1. 必改文件

### `sgl-kernel/include/sgl_kernel_ops.h`

增加自研 C++ 算子声明。

建议覆盖：

- INT8 dynamic activation quant。
- INT8 scaled GEMM。
- INT8 BMM。
- INT8 fused MoE。
- all-reduce / all-gather / reduce-scatter 等通信算子。

### `sgl-kernel/csrc/common_extension.cc`

将自研 C++ 算子注册到 `torch.ops.sgl_kernel.*`。

如果目标后端不是 CUDA，还需要在对应 extension 中注册：

```text
sgl-kernel/csrc/common_extension_rocm.cc
sgl-kernel/csrc/common_extension_musa.cc
sgl-kernel/csrc/cpu/torch_extension_cpu.cpp
```

### `sgl-kernel/CMakeLists.txt`

把自研算子源码加入编译列表。

如果依赖硬件 SDK 或自研通信库，也在这里补充 include、link 和 compile definitions。

### `sgl-kernel/python/sgl_kernel/gemm.py`

增加 Python wrapper，用于调用：

- 自研 INT8 activation quant。
- 自研 INT8 GEMM。
- 自研 INT8 BMM。

这些 wrapper 会被 SGLang 的 quantization method 和 MLA forward 调用。

### `sgl-kernel/python/sgl_kernel/moe.py`

增加自研 fused MoE wrapper。

主要用于替换当前 W8A8 INT8 MoE 中的 Triton fused MoE 路径。

### `sgl-kernel/python/sgl_kernel/allreduce.py`

增加自研通信 wrapper。

至少覆盖：

- all-reduce。
- all-gather。
- reduce-scatter。

## 2. INT8 Linear 路径

### `python/sglang/srt/layers/quantization/w8a8_int8.py`

这是 `DeepSeek-R1-Channel-INT8` dense linear 的核心替换点。

需要修改：

- `W8A8Int8LinearMethod.apply`
  - 将当前 `per_token_quant_int8` 替换为自研 activation quant。
  - 将当前 `int8_scaled_mm` 替换为自研 INT8 GEMM。
- `W8A8Int8MoEMethod.apply`
  - 将当前 Triton MoE runner 替换为自研 fused MoE 算子。

需要注意：

- 保持现有权重和 scale 的加载语义。
- 保持 `ColumnParallelLinear`、`MergedColumnParallelLinear`、`RowParallelLinear` 的 TP 行为。
- 如果自研算子要求特殊权重布局，不建议直接复用原 `w8a8_int8`，应新增独立量化方法。

## 3. 可选：新增独立量化方法

如果目标硬件需要特殊 weight packing、scale layout 或后处理逻辑，建议新增独立量化方法。

### `python/sglang/srt/layers/quantization/kunpeng_w8a8_int8.py`

新增硬件专用 W8A8 INT8 quantization config 和 linear/MoE method。

### `python/sglang/srt/layers/quantization/__init__.py`

注册新的量化方法名称，例如：

```text
kunpeng_w8a8_int8
```

### `python/sglang/srt/server_args.py`

将新量化方法加入 `QUANTIZATION_CHOICES`。

### `python/sglang/srt/configs/model_config.py`

允许新量化方法通过校验。

如需自动识别 `meituan/DeepSeek-R1-Channel-INT8`，也在这里增加保守检测逻辑。

## 4. DeepSeek MLA 路径

### `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`

当前 channel-wise INT8 的 `kv_b_proj` 后处理会把 INT8 权重反量化为 BF16。

如果目标硬件支持 INT8 MLA BMM，需要修改这里：

- 保留 INT8 `kv_b_proj` 权重。
- 拆分得到 INT8 `w_kc` 和 `w_vc`。
- 保存对应 scale。
- 标记当前 attention layer 使用自研 INT8 MLA 路径。

### `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`

将 MLA 中使用 `w_kc` 和 `w_vc` 的 BMM 路径替换为自研 INT8 BMM。

主要替换：

- q-nope 与 `w_kc` 的 BMM。
- attention output 与 `w_vc` 的 BMM。

### `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`

如果实际运行选择 MHA 或 one-shot attention 路径，也需要检查这里是否仍调用未适配的 `kv_b_proj` 或 BMM fallback。

## 5. 通信路径

### `python/sglang/srt/distributed/device_communicators/kunpeng_communicator.py`

新增自研 communicator。

负责封装：

- 通信上下文初始化。
- all-reduce。
- all-gather。
- reduce-scatter。
- 必要的资源释放和错误处理。

### `python/sglang/srt/distributed/parallel_state.py`

在 `GroupCoordinator` 中接入自研 communicator。

需要覆盖：

- 初始化 communicator。
- `all_reduce`。
- `_all_reduce_out_place`。
- `_all_reduce_in_place`。
- `_all_gather_into_tensor`。
- `_reduce_scatter_tensor`。

### `python/sglang/srt/distributed/communication_op.py`

通常不需要大改。

只要 `GroupCoordinator` 内部完成替换，现有 `tensor_model_parallel_all_reduce`、`tensor_model_parallel_all_gather` 等入口可以继续复用。

## 6. MoE Dispatch / Combine

### `python/sglang/srt/layers/moe/token_dispatcher/standard.py`

如果不启用 EP，优先复用 standard dispatcher，只替换 expert compute。

### `python/sglang/srt/layers/moe/token_dispatcher/kunpeng.py`

如果需要 EP all-to-all，新增自研 dispatcher。

负责：

- token dispatch。
- expert output combine。
- expert/token metadata 管理。

### `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

如果新增自研 dispatcher，需要在 `create_moe_dispatcher` 中增加选择逻辑。

第一阶段可以不改这里，先复用 standard dispatcher。

## 7. DeepSeek 模型文件

### `python/sglang/srt/models/deepseek_v2.py`

原则上不大改。

只在以下情况增加少量分支：

- router GEMM 仍强制进入 CUDA 特化路径。
- 某些 DeepSeek 专用 fused op 无法通过 quantization method 替换。
- 需要关闭 FlashInfer、DeepGEMM 或其他未适配 backend。

不要把硬件适配逻辑集中堆到这个文件中。

## 8. 启动方式

如果复用现有 `w8a8_int8`：

```bash
python -m sglang.launch_server \
  --model-path meituan/DeepSeek-R1-Channel-INT8 \
  --quantization w8a8_int8 \
  --trust-remote-code \
  --tp-size <N>
```

如果新增独立量化方法：

```bash
python -m sglang.launch_server \
  --model-path meituan/DeepSeek-R1-Channel-INT8 \
  --quantization kunpeng_w8a8_int8 \
  --trust-remote-code \
  --tp-size <N>
```

## 9. 最小文件清单

第一阶段最少关注：

```text
sgl-kernel/include/sgl_kernel_ops.h
sgl-kernel/csrc/common_extension.cc
sgl-kernel/CMakeLists.txt
sgl-kernel/python/sgl_kernel/gemm.py
sgl-kernel/python/sgl_kernel/moe.py
sgl-kernel/python/sgl_kernel/allreduce.py
python/sglang/srt/layers/quantization/w8a8_int8.py
python/sglang/srt/distributed/device_communicators/kunpeng_communicator.py
python/sglang/srt/distributed/parallel_state.py
python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
```

如果新增独立量化方法，再加：

```text
python/sglang/srt/layers/quantization/kunpeng_w8a8_int8.py
python/sglang/srt/layers/quantization/__init__.py
python/sglang/srt/server_args.py
python/sglang/srt/configs/model_config.py
```

如果需要 EP dispatch/combine，再加：

```text
python/sglang/srt/layers/moe/token_dispatcher/kunpeng.py
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```
