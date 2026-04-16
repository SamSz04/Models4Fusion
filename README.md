# Models4Fusion
Model zoo for generating Hlo dataset and further experiment on RL fusion.

## XLA Flags
```
# 设置 XLA Flags:
# --xla_dump_to: 输出路径
# --xla_dump_hlo_as_text: 生成 .txt
# --xla_dump_hlo_as_dot: 生成 .dot (可视化)
# --xla_dump_hlo_as_proto: 生成 .pb (二进制)
# --xla_dump_hlo_pass_re=.*: 导出所有阶段的图 (包括优化前和优化后)
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=./xla_dumps_covdot_4080_260416 "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_html "
    # "--xla_dump_hlo_as_proto "
    "--xla_dump_hlo_pass_re=.* "
    # "--xla_dump_hlo_pipeline_re=.*fusion.* "
    # "--xla_dump_hlo_pass_re=.*fusion.* "
    "--xla_dump_hlo_module_re=simple_computation "
    "--xla_dump_fusion_visualization=true "
    # "--xla_gpu_collect_cost_model_stats=true "
)
```

## Commands
```
./bazel-bin/xla/tools/run_hlo_module \
  --platform=CUDA --reference_platform="" \
  --xla_disable_hlo_passes=layout-assignment, layout_normalization \
  --xla_dump_to="$OUT/{greedy|beam}" \
  --xla_dump_hlo_pass_re='.*' \
  --xla_dump_hlo_as_text=true \
  --xla_dump_hlo_as_html=true \
  --xla_gpu_priority_fusion_beam_width=<1 or 8> \
  --xla_gpu_priority_fusion_beam_depth=<0 or 64> \
  /root/hlo_dumps/mlp_gpt/0060before_fusion.hlo
```
