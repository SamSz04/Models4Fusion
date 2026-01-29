import os
import time
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

# 尝试导入 Pallas GPU 后端
try:
    from jax.experimental.pallas import gpu as plgpu
except ImportError:
    plgpu = None
    print("警告: 未检测到 Pallas GPU 支持。代码将在 CPU 上模拟或报错。")

# =============================================================================
# 第一部分：环境配置与 HLO 内省设置
# =============================================================================
# 必须在执行任何 JAX 计算前设置。这将指示 XLA 编译器将中间表示转储到指定目录。
# 对应的 Research Snippet: [4, 8]

# 创建转储目录
dump_dir = "../test_layout_tile/hlo_dumps_nchw"
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={dump_dir} "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_dot "
)


# =============================================================================
# 第二部分：标准 JAX 实现 (用于生成 HLO 与 布局配置实验)
# =============================================================================

def standard_computation(input_tensor, kernel, weights, layout_config='NHWC'):
    """
    包含 Conv, Activation, Dot 的复合函数。
    支持通过 layout_config 参数改变卷积布局，从而观察 HLO 的变化。
    """

    # 根据配置选择维度编号
    # NHWC (默认, Tensor Core 友好): ('NHWC', 'HWIO', 'NHWC')
    # NCHW (传统): ('NCHW', 'OIHW', 'NCHW')
    if layout_config == 'NHWC':
        # input: NHWC, kernel: HWIO, output: NHWC
        dnums = ('NHWC', 'HWIO', 'NHWC')
    else:
        # input: NCHW, kernel: OIHW, output: NCHW
        dnums = ('NCHW', 'OIHW', 'NCHW')

    dn = lax.conv_dimension_numbers(
        input_tensor.shape,
        kernel.shape,
        dnums
    )

    # 1. 卷积
    conv_out = lax.conv_general_dilated(
        lhs=input_tensor,
        rhs=kernel,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=dn
    )

    # 2. 激活
    activation = jax.nn.relu(conv_out)

    # 3. 矩阵乘法准备
    # 需要根据布局正确展平数据
    if layout_config == 'NHWC':
        b, h, w, c = activation.shape
        # ->
        flattened = activation.reshape((b * h * w, c))
    else:  # NCHW
        b, c, h, w = activation.shape
        # -> ->
        # 注意：这里需要 transpose，XLA 可能会将其与上面的卷积输出融合或插入 copy
        transposed = activation.transpose((0, 2, 3, 1))
        flattened = transposed.reshape((b * h * w, c))

    # 4. 矩阵乘法
    projected = lax.dot(flattened, weights)

    return projected


# =============================================================================
# 第三部分：Pallas 实现 (用于显式 Tile Size 配置与调优)
# =============================================================================
# Research Snippet Reference: [7, 9, 10, 11]

def make_pallas_matmul_kernel(block_m, block_n, block_k):
    """
    生成一个具有固定 Tile Size 的 Pallas 矩阵乘法内核。

    参数:
      x_ref: 输入矩阵 A 的引用 [M, K]
      y_ref: 输入矩阵 B 的引用 [K, N]
      z_ref: 输出矩阵 C 的引用 [M, N]
    """

    def matmul_kernel(x_ref, y_ref, z_ref):
        # 1. 初始化累加器 (Accumulator)
        # 这通常存储在寄存器或高速 SMEM 中
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)

        # 获取 K 维度的大小
        k_size = x_ref.shape[1]

        # 2. K 维循环 (Main Loop)
        # 每次迭代加载一个 block_k 大小的块
        def body(i, acc):
            # 计算当前 K 维度的偏移量
            start_k = i * block_k

            # 从全局内存 (Global Memory) 加载数据块到引用 (Ref)
            # pl.dslice (或简单的切片语法) 用于定义加载窗口
            # 这里的切片操作对应于从 HBM 到 SRAM 的数据传输
            x_tile = x_ref[:, start_k: start_k + block_k]
            y_tile = y_ref[start_k: start_k + block_k, :]

            # 计算并累加 (在 Tensor Core 或 CUDA Core 上执行)
            return acc + jnp.dot(x_tile, y_tile)

        # 使用 lax.fori_loop 进行循环，这是编译为 GPU 循环的标准方式
        acc = lax.fori_loop(0, k_size // block_k, body, acc)

        # 3. 写回结果
        # 将计算结果从累加器写回全局内存
        z_ref[:, :] = acc

    return matmul_kernel


def pallas_tiled_matmul(x, y, bm=128, bn=128, bk=32):
    """
    执行具有显式分块配置的矩阵乘法。
    """
    m, k = x.shape
    _, n = y.shape

    # 确保维度能被分块整除 (为了简化代码，不做边界检查)
    assert m % bm == 0 and n % bn == 0 and k % bk == 0

    # 定义网格 (Grid): 启动多少个并行程序实例
    grid = (m // bm, n // bn)

    # 定义 BlockSpecs: 网格索引如何映射到数据块
    # 这里的 lambda 函数定义了每个程序实例 (i, j) 负责的数据范围
    # 输入 A: 读取 [bm, k] (注意：虽然内核内部循环 K，但这里 spec 声明整个 K 范围是可访问的，
    # 或者我们可以更精细地控制。为简化，允许内核访问整行/整列)
    in_specs = [
        # x (A矩阵): 映射到 Grid 的第 i 行块，但保留完整的 k 维度
        pl.BlockSpec(lambda i, j: (i, 0), (bm, k)),

        # y (B矩阵): 映射到 Grid 的第 j 列块，但保留完整的 k 维度
        pl.BlockSpec(lambda i, j: (0, j), (k, bn))
    ]

    out_specs = pl.BlockSpec(lambda i, j: (i, j), (bm, bn))

    # 编译并调用 Pallas 内核
    return pl.pallas_call(
        make_pallas_matmul_kernel(bm, bn, bk),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid
    )(x, y)


# =============================================================================
# 第四部分：基准测试与运行
# =============================================================================

def run_benchmark():
    print("# JAX/XLA 优化研究：布局与分块策略")
    print("--------------------------------------------------")

    # 1. 准备数据
    B, H, W, C = 4, 64, 64, 128
    K_H, K_W, C_OUT = 3, 3, 256
    PROJ_DIM = 512

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # 对于 Standard JAX (Layout 实验)
    # NHWC 格式输入
    input_nhwc = jax.random.normal(k1, (B, H, W, C))
    # NCHW 格式输入 (用于对比)
    input_nchw = input_nhwc.transpose(0, 3, 1, 2)

    kernel_hwio = jax.random.normal(k2, (K_H, K_W, C, C_OUT))
    kernel_oihw = kernel_hwio.transpose(3, 2, 0, 1)

    proj_weights = jax.random.normal(k3, (C_OUT, PROJ_DIM))

    # 2. 运行布局配置实验 (Layout)
    print(f"\n[实验 A] 布局配置 (Layout) 对比与 HLO 生成")
    print(f"HLO 文件将生成于: {os.path.abspath(dump_dir)}")

    # # Case 1: NHWC (推荐)
    # print("  正在编译与运行 NHWC 配置...")
    # fn_nhwc = jax.jit(functools.partial(standard_computation, layout_config='NHWC'))
    # # 以此触发编译和 HLO dump
    # _ = fn_nhwc(input_nhwc, kernel_hwio, proj_weights).block_until_ready()

    # Case 2: NCHW (非推荐，预期会产生 Copy)
    print("  正在编译与运行 NCHW 配置...")
    fn_nchw = jax.jit(functools.partial(standard_computation, layout_config='NCHW'))
    # 以此触发编译和 HLO dump
    _ = fn_nchw(input_nchw, kernel_oihw, proj_weights).block_until_ready()

    print("  >>> 请检查 dump 目录，寻找 'transpose' 或 'copy' 指令的差异。")

    # 3. 运行分块配置实验 (Tile Size via Pallas)
    print(f"\n 分块大小 (Tile Size) 性能基准测试")
    # 模拟矩阵乘法部分的数据
    M = B * H * W  # 4*64*64 = 16384
    K = C_OUT  # 256
    N = PROJ_DIM  # 512

    mat_a = jax.random.normal(key, (M, K))
    mat_b = jax.random.normal(key, (K, N))

    print(f"  矩阵规模: [{M}, {K}] x [{K}, {N}]")

    # 定义要测试的配置列表 (BM, BN, BK)
    configs = [
        (64, 64, 32),  # 保守配置 (Safe baseline)
        (128, 64, 32),  # 增加 M 维度的并行粒度
        (64, 128, 32),  # 增加 N 维度的并行粒度
        (128, 128, 32),  # 较大的 Tile (高性能，但对 SRAM 要求高)
        (256, 32, 32)  # 极宽的 M (适合 M >> N 的情况)
    ]

    for bm, bn, bk in configs:
        config_name = f"Tile[{bm}x{bn}x{bk}]"
        try:
            # 编译特定的 Tile 配置
            # 注意: Pallas 内核在第一次调用时会被编译
            tiled_fn = jax.jit(functools.partial(pallas_tiled_matmul, bm=bm, bn=bn, bk=bk))

            # Warmup
            _ = tiled_fn(mat_a, mat_b).block_until_ready()

            # Benchmark
            start_time = time.time()
            iters = 100
            for _ in range(iters):
                _ = tiled_fn(mat_a, mat_b).block_until_ready()
            end_time = time.time()

            avg_ms = (end_time - start_time) / iters * 1000
            print(f"  配置 {config_name}: 平均耗时 = {avg_ms:.4f} ms")

        except Exception as e:
            print(f"  配置 {config_name}: 失败 - {str(e)}")
            print("  (注: 失败通常是由于 Tile Size 超过了硬件的 Shared Memory 限制)")


if __name__ == "__main__":
    run_benchmark()