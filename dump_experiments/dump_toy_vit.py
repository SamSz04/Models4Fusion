import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

# ==========================================
# 1. 设置 XLA Dump 环境变量 (最关键步骤)
# ==========================================
dump_dir = "./hlo_mini_vit"
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

# 设置 Flags 以导出所有格式
# --xla_dump_hlo_pass_re=.*: 导出所有优化阶段的图
flags = (
    f"--xla_dump_to={dump_dir} "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_dot "
    "--xla_dump_hlo_as_proto "
)
os.environ["XLA_FLAGS"] = flags
print(f"XLA Flags 已设置: {flags}")


# ==========================================
# 2. 定义一个极简的 Vision Transformer
# ==========================================
class MiniViT(nn.Module):
    patch_size: int = 4  # 以此划分图片，例如 32x32 -> 8x8 个 patches
    embed_dim: int = 64  # 极小的维度，方便看图
    num_heads: int = 2  # 只有2个头
    num_layers: int = 2  # 只有2层 Transformer Block
    num_classes: int = 10  # 输出类别数

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x shape: [Batch, Height, Width, Channels]

        # 1. Patch Embedding (使用 Conv2D 实现，这是最高效的方式)
        # 结果 shape: [Batch, H/patch, W/patch, embed_dim]
        x = nn.Conv(features=self.embed_dim,
                    kernel_size=(self.patch_size, self.patch_size),
                    strides=(self.patch_size, self.patch_size),
                    padding='VALID')(x)

        b, h, w, c = x.shape
        # Flatten: [Batch, Sequence_Length, Embed_Dim]
        x = x.reshape((b, h * w, c))

        # 2. 添加 Position Embedding
        # 创建可学习的位置编码参数
        num_patches = h * w
        pos_emb = self.param('pos_emb',
                             nn.initializers.normal(stddev=0.02),
                             (1, num_patches, self.embed_dim))
        x = x + pos_emb

        # 3. Transformer Encoder Blocks (循环极少次)
        for _ in range(self.num_layers):
            # Layer Norm 1
            y = nn.LayerNorm()(x)
            # Self Attention
            y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
                                                kernel_init=nn.initializers.xavier_uniform())(y, y)
            x = x + y  # Residual

            # Layer Norm 2
            y = nn.LayerNorm()(x)
            # MLP
            y = nn.Dense(features=self.embed_dim * 2)(y)  # 这里的扩展倍数通常是4，为了减小图我们设为2
            y = nn.gelu(y)
            y = nn.Dense(features=self.embed_dim)(y)
            x = x + y  # Residual

        # 4. Classification Head
        x = nn.LayerNorm()(x)
        x = x.mean(axis=1)  # Global Average Pooling (把序列维度平均掉)
        x = nn.Dense(features=self.num_classes)(x)

        return x


# ==========================================
# 3. 准备数据和初始化
# ==========================================
# 模拟一张极小的图片输入 (Batch=1, 32x32 RGB)
# 这样生成的计算图不会包含大量的 Padding 或切片操作
input_shape = (1, 32, 32, 3)
dummy_input = jnp.ones(input_shape, dtype=jnp.float32)

model = MiniViT()
key = jax.random.PRNGKey(0)

print("正在初始化模型参数...")
variables = model.init(key, dummy_input)

# ==========================================
# 4. JIT 编译并触发 Dump
# ==========================================
print("正在 JIT 编译并运行...")


@jax.jit
def forward_pass(params, inputs):
    return model.apply(params, inputs)


# 第一次运行会触发 XLA 编译 -> 生成 HLO 文件
output = forward_pass(variables, dummy_input)
# 强制等待计算完成
output.block_until_ready()

print(f"运行完成！输出 Shape: {output.shape}")
print(f"请检查目录: {os.path.abspath(dump_dir)}")
print("-" * 30)
print("提示：在生成的文件夹中，寻找文件名包含 'optimized' 的 .txt 或 .pb 文件。")
print("由于模型很小，你可以尝试用 Graphviz 打开 .dot 文件查看可视化结构。")