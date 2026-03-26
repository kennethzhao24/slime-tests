# Slime-tests

**slime** is an LLM post-training framework for RL scaling, providing two core capabilities:

1.  **High-Performance Training**: Supports efficient training in various modes by connecting Megatron with SGLang;
2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

# TO-DO-List:
- [x] Data/Model
  - [x] Dataset Prep
  - [x] Convert to slime Format
- [ ] GRPO
  - [x] [1N4G](scripts/run-qwen3-4B.sh)
  - [x] [2N8G](scripts/run-qwen3-4B-2N8G.sh)
  - [ ] [4N16G](scripts/run-qwen3-4B-4N16G.sh)

### 1. Run the container 
```bash
bash scripts/run_apptainer.sh
```

If the default image is incompatible with the node's NVIDIA driver, point the launcher at a different image:
```bash
APPTAINER_IMAGE=/path/to/compatible.sif bash scripts/run_apptainer.sh
```

## Example: Run Qwen3-4B with GRPO on DAPO-MATH (Single GPU)

### 1. Convert models to megatron format
```bash
bash scripts/convert_hf_megatron.sh
```

### 2. Run GRPO on Qwen3-4B
```bash
bash scripts/run-qwen3-4B.sh 2>&1 | tee run.log
```
