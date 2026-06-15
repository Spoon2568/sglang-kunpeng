#!/bin/bash
# Kunpeng SGLang 完整测试套件
#
# 运行所有 Kunpeng 相关的单元测试和集成测试
#
# Usage:
#   source ~/sibow/init.sh
#   cd /home/share/fengguangnan/sibow/llminfer/sglang-kunpeng
#   bash scripts/run_kunpeng_tests.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 环境检查
echo "======================================================================"
echo "Kunpeng SGLang 测试套件"
echo "======================================================================"
echo ""

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ 虚拟环境已激活"
else
    echo -e "${RED}✗ 未找到虚拟环境 .venv/bin/activate${NC}"
    exit 1
fi

# 检查 SO 文件
if [ -z "$KUNPENG_ASYNC_COMPUTE_SO" ]; then
    export KUNPENG_ASYNC_COMPUTE_SO="/home/share/fengguangnan/sibow/llminfer/Kpllminfer/kernels/async_compute_op.so"
fi

if [ ! -f "$KUNPENG_ASYNC_COMPUTE_SO" ]; then
    echo -e "${RED}✗ 未找到 async_compute_op.so: $KUNPENG_ASYNC_COMPUTE_SO${NC}"
    exit 1
fi
echo "✓ SO 文件: $KUNPENG_ASYNC_COMPUTE_SO"

# 设置环境变量
export SGLANG_USE_KUNPENG_TP=1
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=16

echo ""
echo "======================================================================"
echo "单机测试（不需要 MPI）"
echo "======================================================================"
echo ""

SINGLE_TESTS=(
    "test/kunpeng_workspace_lifetime_test.py"
    "test/kunpeng_moe_tilebuf_test.py"
    "test/kunpeng_mla_bmm_cscale_test.py"
    "test/kunpeng_operators_test.py"
    "test/kunpeng_topk_test.py"
    "test/kunpeng_moe_int8_test.py"
    "test/kunpeng_mla_int8_bmm_test.py"
)

SINGLE_PASSED=0
SINGLE_FAILED=0

for test in "${SINGLE_TESTS[@]}"; do
    if [ -f "$test" ]; then
        echo "----------------------------------------------------------------------"
        echo "运行: $test"
        echo "----------------------------------------------------------------------"
        if python "$test"; then
            echo -e "${GREEN}✓ PASS${NC}: $test"
            SINGLE_PASSED=$((SINGLE_PASSED + 1))
        else
            echo -e "${RED}✗ FAIL${NC}: $test"
            SINGLE_FAILED=$((SINGLE_FAILED + 1))
        fi
        echo ""
    else
        echo -e "${YELLOW}⊘ SKIP${NC}: $test (文件不存在)"
    fi
done

echo ""
echo "======================================================================"
echo "分布式测试（需要 MPI）"
echo "======================================================================"
echo ""

# 检查 mpirun
if ! command -v mpirun &> /dev/null; then
    echo -e "${YELLOW}⊘ 跳过分布式测试: 未找到 mpirun${NC}"
    DIST_TESTS=()
else
    DIST_TESTS=(
        "4:test/kunpeng_lifecycle_test.py"
        "8:test/kunpeng_tp_consistency_test.py"
        "8:test/kunpeng_allgather_shape_test.py"
    )
fi

DIST_PASSED=0
DIST_FAILED=0

for test_spec in "${DIST_TESTS[@]}"; do
    IFS=':' read -r np test <<< "$test_spec"

    if [ -f "$test" ]; then
        echo "----------------------------------------------------------------------"
        echo "运行: mpirun -np $np python $test"
        echo "----------------------------------------------------------------------"

        # 为每个测试分配不同的 MASTER_PORT 避免冲突
        port=$((29500 + $(echo "$test" | md5sum | cut -c1-4 | tr 'a-f' '0-5')))

        if MASTER_PORT=$port mpirun -np $np python "$test"; then
            echo -e "${GREEN}✓ PASS${NC}: $test (np=$np)"
            DIST_PASSED=$((DIST_PASSED + 1))
        else
            echo -e "${RED}✗ FAIL${NC}: $test (np=$np)"
            DIST_FAILED=$((DIST_FAILED + 1))
        fi
        echo ""
    else
        echo -e "${YELLOW}⊘ SKIP${NC}: $test (文件不存在)"
    fi
done

echo ""
echo "======================================================================"
echo "测试总结"
echo "======================================================================"
echo ""

TOTAL_PASSED=$((SINGLE_PASSED + DIST_PASSED))
TOTAL_FAILED=$((SINGLE_FAILED + DIST_FAILED))
TOTAL_TESTS=$((TOTAL_PASSED + TOTAL_FAILED))

echo "单机测试: $SINGLE_PASSED passed, $SINGLE_FAILED failed"
echo "分布式测试: $DIST_PASSED passed, $DIST_FAILED failed"
echo ""
echo "总计: $TOTAL_PASSED/$TOTAL_TESTS passed"
echo ""

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}======================================================================"
    echo "✓ 所有测试通过！"
    echo "======================================================================${NC}"
    exit 0
else
    echo -e "${RED}======================================================================"
    echo "✗ $TOTAL_FAILED 个测试失败"
    echo "======================================================================${NC}"
    exit 1
fi
