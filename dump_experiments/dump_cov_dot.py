import os
import shutil

# =============================================================================
# 1. 配置 XLA Dump 环境变量 (必须在 import jax 之前完成)
# =============================================================================
output_dir = "./hlo_dot_cov_dump_layout"

# 如果目录存在，先清理以便观察新生成的文件
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# 设置 XLA Flags:
# --xla_dump_to: 输出路径
# --xla_dump_hlo_as_text: 生成 .txt
# --xla_dump_hlo_as_dot: 生成 .dot (可视化)
# --xla_dump_hlo_as_proto: 生成 .pb (二进制)
# --xla_dump_hlo_pass_re=.*: 导出所有阶段的图 (包括优化前和优化后)
flags = (
    f"--xla_dump_to={output_dir} "
    # "--xla_dump_hlo_as_text "
    # "--xla_dump_hlo_as_dot "
    # "--xla_dump_hlo_as_proto "
    "--xla_dump_hlo_pass_re=layout "
)
os.environ["XLA_FLAGS"] = flags
print(f"XLA Flags 设置为: {flags}")

# =============================================================================
# 2. 导入 JAX
# =============================================================================
import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# 3. 定义计算图
# =============================================================================
def simple_computation(img, kernel, weights, bias):
    """
    包含: Convolution -> Reshape -> Dot -> Add
    """

    # --- 1. Convolution ---
    # img: [Batch, Height, Width, Channel] -> NHWC
    # kernel: [K_Height, K_Width, In_Channel, Out_Channel] -> HWIO
    # window_strides=(1, 1): 步长
    # padding='VALID': 不填充
    # dimension_numbers: 指定数据布局，确保生成标准的卷积
    conv_out = lax.conv_general_dilated(
        lhs=img,
        rhs=kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    # 假设输入是 [1, 8, 8, 2], Kernel是 [3, 3, 2, 4]
    # 输出将是 [1, 6, 6, 4]

    # --- 2. Reshape ---
    # 将卷积输出展平 [Batch, H*W*C]
    # 1 * 6 * 6 * 4 = 144
    batch_size = conv_out.shape[0]
    flattened = jnp.reshape(conv_out, (batch_size, -1))

    # --- 3. Dot (Matrix Multiplication) ---
    # weights: [144, 10]
    # flattened: [1, 144]
    # dot_out: [1, 10]
    dot_out = jnp.dot(flattened, weights)

    # --- 4. Add ---
    # bias: [1, 10]
    final_out = dot_out + bias

    return final_out


# =============================================================================
# 4. 准备数据并触发 JIT 编译
# =============================================================================

# 定义维度
B, H, W, C_in = 1, 8, 8, 2
K_H, K_W, C_out = 3, 3, 4
Dense_Out = 10

# 随机初始化数据
key = jax.random.PRNGKey(0)
k1, k2, k3, k4 = jax.random.split(key, 4)

img = jax.random.normal(k1, (B, H, W, C_in))  # [1, 8, 8, 2]
kernel = jax.random.normal(k2, (K_H, K_W, C_in, C_out))  # [3, 3, 2, 4]

# 计算一下 Flatten 后的维度: (8-3+1) * (8-3+1) * 4 = 6*6*4 = 144
flat_dim = (H - K_H + 1) * (W - K_W + 1) * C_out
weights = jax.random.normal(k3, (flat_dim, Dense_Out))  # [144, 10]
bias = jax.random.normal(k4, (1, Dense_Out))  # [1, 10]

print("准备 JIT 编译...")

# 使用 @jax.jit 装饰器 (或者直接调用 jax.jit) 来触发 XLA 编译
# 只有在编译发生时，HLO 才会 dump 到硬盘
jit_fn = jax.jit(simple_computation)

print("正在运行计算...")
result = jit_fn(img, kernel, weights, bias)

# 强制等待计算完成
result.block_until_ready()

print(f"运行完成。输出 Shape: {result.shape}")
print("-" * 30)
print(f"文件已生成至: {os.path.abspath(output_dir)}")
print("请寻找包含 'optimized' 关键字的文件，例如:")
print("module_xxxx.jit_simple_computation.cpu_after_optimizations.txt (或 .pb)")