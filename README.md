# Slime-tests

**slime** is an LLM post-training framework for RL scaling, providing two core capabilities:

1.  **High-Performance Training**: Supports efficient training in various modes by connecting Megatron with SGLang;
2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

# TO-DO-List:
- [x] Data/Model
  - [x] Dataset Prep
  - [x] Convert to slime Format
- [x] GRPO
  - [x] [1N4G](scripts/run-qwen3-4B.sh)
  - [x] [2N8G](scripts/run-qwen3-4B-2N8G.sh)
  - [x] [4N16G](scripts/run-qwen3-4B-4N16G.sh)
  - [x] [xN4xG generic](scripts/run-qwen3-4B-xN4xG.sh)
- [ ] PD Rollout
- [ ] Non-colocated



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

## Generic multi-node launcher

Use the generic launcher when you want the same Qwen3-4B GRPO setup on `x`
nodes with 4 GPUs per node:

```bash
ACTOR_NUM_NODES=2 MASTER_ADDR=<head_ip> LOCAL_NODE_IP=<head_ip> SOCKET_IFNAME=hsn0 ROLE=head \
  bash scripts/run-qwen3-4B-xN4xG.sh
```

The worker nodes should run the same script with `ROLE=worker` and their own
`LOCAL_NODE_IP`.
