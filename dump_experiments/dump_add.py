import os

# 设置 XLA flag
# --xla_dump_to: 指定输出目录
# --xla_dump_hlo_as_dot: 导出为 .dot 格式 (Graphviz)
# os.environ["XLA_FLAGS"] = "--xla_dump_to=./xla_dumps_tile --xla_dump_hlo_as_dot --xla_dump_hlo_as_text"
os.environ["XLA_FLAGS"] = "--xla_dump_to=./xla_dumps_tile --xla_dump_hlo_pass_re=tile"

import jax
import jax.numpy as jnp

# 你的计算代码
def add_matrices(a, b):
    return a + b

a = jax.ShapeDtypeStruct((128, 64), jnp.float32)
b = jax.ShapeDtypeStruct((128, 64), jnp.float32)
lowered = jax.jit(add_matrices).lower(a, b)
# 触发编译，这不仅会生成 HLO，还会触发 XLA pass 并生成 dump 文件
compiled = lowered.compile()