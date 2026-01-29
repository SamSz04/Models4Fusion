import os
import sys

# ==========================================
# 1. 设置 XLA Dump 环境变量 (必须在 import jax 之前)
# ==========================================
# 指定 dump 文件的保存路径
dump_dir = "xla_dump_mlp_gpt_jax"
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

# 设置 XLA Flags:
# --xla_dump_to: 输出目录
# --xla_dump_hlo_as_text: 输出 .txt 格式
# --xla_dump_hlo_as_dot: 输出 .dot 格式 (可视化)
# --xla_dump_hlo_as_proto: 输出 .pb 格式 (Protobuf)
# --xla_dump_hlo_pass_re=.*: 正则表达式，匹配所有 pass (建议加上，否则可能只输出部分)
flags = (
    f"--xla_dump_to={dump_dir} "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_dot "
    "--xla_dump_hlo_as_proto "
    "--xla_dump_hlo_pass_re=.*"
)

os.environ["XLA_FLAGS"] = flags
print(f"XLA Flags set to: {os.environ['XLA_FLAGS']}")

# ==========================================
# 2. 导入 JAX 和模型库
# ==========================================
import jax
import jax.numpy as jnp
from jax import random
from haiku import PRNGSequence

# 假设 mlp_gpt_jax 在当前路径或 pythonpath 下
try:
    from mlp_gpt_jax import TransformedMLPGpt
except ImportError:
    print("错误: 无法导入 'mlp_gpt_jax'。请确保该文件在你的 Python 路径中。")
    sys.exit(1)

# ==========================================
# 3. 检查 TPU 设备
# ==========================================
# print(f"JAX 平台: {jax.lib.xla_bridge.get_backend().platform}")
devices = jax.devices()
print(f"可用设备: {devices}")

# if jax.lib.xla_bridge.get_backend().platform != 'tpu':
#     print("警告: 当前未检测到 TPU 后端，将在 CPU/GPU 上生成 HLO (格式相同，但针对的硬件优化不同)。")

# ==========================================
# 4. 定义模型与输入 (使用你提供的代码)
# ==========================================
model = TransformedMLPGpt(
    num_tokens=20000,
    dim=512,
    depth=6,
    seq_len=1024
)

rng = PRNGSequence(0)
# 注意：确保输入数据的 shape 和 dtype 是确定的
seq = random.randint(next(rng), (1024,), 0, 20000)

print("正在初始化参数...")
# 初始化通常不需要 JIT，除非模型非常巨大
params = model.init(next(rng), seq)

# ==========================================
# 5. 核心步骤：使用 jax.jit 触发 XLA 编译
# ==========================================
# HLO 只有在编译阶段才会生成。如果不使用 @jax.jit，
# 代码会以 Eager 模式（逐行解释）运行，不会产生完整的计算图 dump。

@jax.jit
def forward_pass(params, rng_key, seq):
    # 这里调用模型的 apply 函数
    return model.apply(params, rng_key, seq)

print("正在运行并触发 JIT 编译 (这将生成 HLO dump)...")

# 第一次运行会触发编译 (Tracing -> HLO -> Binary)
# 此时你应该会在 ./xla_dump_mlp_gpt_jax 目录下看到文件生成
logits = forward_pass(params, next(rng), seq)

# block_until_ready() 强制等待异步执行完成，确保 dump 完整写入
logits.block_until_ready()

print(f"运行完成! 输出 Logits Shape: {logits.shape}")
print(f"请检查目录 '{dump_dir}' 查看生成的 HLO 文件。")
print("通常你需要寻找文件名包含 'thunk' 或 'optimized' 的文件来分析最终图。")