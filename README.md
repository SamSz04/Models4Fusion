# Models4Fusion
Model zoo for generating Hlo dataset and further experiment on RL fusion.


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
