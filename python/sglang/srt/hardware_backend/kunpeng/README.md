# Kunpeng Hardware Backend for SGLang

Kunpeng 硬件后端适配，用于在鲲鹏 ARM CPU 超算集群上运行 DeepSeek-R1 Channel-INT8 量化模型。

## 架构

- **INT8 Linear**: 使用 `async_compute` 的 `igemm_bdq` INT8 GEMM 算子
- **INT8 MoE**: 使用 `igemm_fusedmoe_gateup/down` 融合 MoE 算子
- **MLA Batched GEMM**: 使用 `batched_gemm_woqs8` INT8 batched GEMM
- **RMSNorm**: 使用 `rmsnorm_out` / `add_rmsnorm_out` 融合 norm 算子
- **Grouped TopK**: 使用 `grouped_topk_out` 路由算子

## 环境变量

启动前必须设置以下环境变量：

```bash
# 启用 Kunpeng W8A8 后端
export SGLANG_USE_KUNPENG_W8A8=1

# 指向你的 async_compute 算子库路径
export KUNPENG_ASYNC_COMPUTE_SO=/path/to/Kpllminfer/kernels/async_compute_op.so
```

## 启动命令

```bash
export SGLANG_USE_KUNPENG_W8A8=1
export KUNPENG_ASYNC_COMPUTE_SO=/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so

python -m sglang.launch_server \
  --model-path meituan/DeepSeek-R1-Channel-INT8 \
  --quantization w8a8_int8 \
  --disable-shared-experts-fusion \
  --trust-remote-code \
  --tp-size 1
```

## 重要：Shared Expert 模式

Kunpeng 后端使用**模式 B（独立模式）**处理共享专家，与 KunPengDistInfer 架构一致：

- Routed experts: 通过 TopK 路由，MoE kernel 计算
- Shared expert: 独立的 MLP 路径计算，不参与 TopK

**必须**使用 `--disable-shared-experts-fusion` 启动，否则会报错：
```
RuntimeError: Kunpeng W8A8 backend does not support fused shared experts (mode A).
Shared experts must be computed independently (mode B).
Please launch the server with --disable-shared-experts-fusion
```

## 模块结构

```
python/sglang/srt/hardware_backend/kunpeng/
├── quantization/
│   └── w8a8_int8.py       # INT8 量化算子封装 (Linear, MoE, MLA BMM)
├── norm.py                 # RMSNorm 算子封装
├── topk.py                 # Grouped TopK 路由算子封装
└── README.md               # 本文件
```

## 与 KunPengDistInfer 的对应关系

| SGLang 模块 | KunPengDistInfer | 说明 |
|------------|------------------|------|
| `apply_linear` | `igemm_bdq` | INT8 linear GEMM |
| `apply_moe` | `igemm_fusedmoe_gateup/down` | INT8 MoE 融合算子 |
| `_batched_gemm_uk/uv` | `batched_gemm_woqs8` | MLA INT8 batched GEMM |
| `rmsnorm_forward_kunpeng` | `rmsnorm_out` / `add_rmsnorm_out` | RMSNorm |
| `grouped_topk_forward` | `grouped_topk` | Grouped TopK 路由 |

## 限制

1. **必须使用 `--disable-shared-experts-fusion`**
2. 目前只支持 `w8a8_int8` 量化方法
3. 需要预编译的 `async_compute_op.so` 算子库
4. MLA batched GEMM 需要权重为 channel-wise INT8

## 开发者参考

详细的适配指南请参考：
- [docs/developer_guide/deepseek_r1_channel_int8_custom_backend.md](../../../docs/developer_guide/deepseek_r1_channel_int8_custom_backend.md)
