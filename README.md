# Global Humanoid Robot Challenge 2026 Baseline

This is the official technical documentation for Global Humanoid Robot Challenge 2026 (GHRC 2026). It is built on the [LeRobot](https://github.com/huggingface/lerobot) framework and provides an end-to-end workflow for a humanoid robot simulation platform, covering physics simulation, data collection, model training, and deployment.

## Project Overview

This documentation is intended for **GHRC 2026** participants and R&D teams, providing a unified baseline implementation:

- Build a high-fidelity humanoid robot simulation environment based on **NVIDIA Isaac Sim**
- Collect data via keyboard teleoperation and export in the standardized **LeRobotDataset V3.0** format
- Train and fine-tune models using imitation learning algorithms (e.g., ACT, SmolVLA, Pi0)

## Key Capabilities

| Capability | Description |
| :--: | --- |
| **Simulation Environment** | High-fidelity Walker S2 robot simulation based on NVIDIA Isaac Sim, supporting a 20-dimensional state space (14 arm joints + 4 gripper joints + 2 gripper control commands). |
| **Data Collection** | Supports keyboard teleoperation; exports **LeRobotDataset V3.0** format. |
| **Model Training** | Supports imitation learning algorithms such as **ACT** and **Pi0**. |
| **4-View Real-time Display** | Supports real-time preview from 4 RGB cameras (head_left, head_right, wrist_left, wrist_right). |

## Resources

Some large files in this project are hosted on Hugging Face. **Please download them before first use**:

| Resource Type | Local Directory | Remote |
| --- | --- | --- |
| 🤖 Simulation environment & robot assets | `assets/` (Git submodule) | [UBTECH-Robotics/challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets) |
| 📊 Training dataset | `datasets/` | [UBTECH-Robotics/challenge2026_dataset](https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset) |

### Recommended Configuration

|  | Minimum | Recommended | Ideal |
| --- | --- | --- | --- |
| **OS** | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 |
| **CPU** | Intel Core i7 (7th Gen); AMD Ryzen 5 | Intel Core i7 (9th Gen); AMD Ryzen 7 | Intel Core i9 (X-series or higher); AMD Ryzen 9 / Threadripper (or higher) |
| **Cores** | 4 | 8 | 16 |
| **RAM** | 32GB | 64GB | 64GB |
| **Storage** | 50GB SSD | 500GB SSD | 1TB NVMe SSD |
| **GPU** | GeForce RTX 4080 | GeForce RTX 5080 | RTX PRO 6000 Blackwell |
| **VRAM** | 16GB | 16GB | 48GB |
| **Driver** | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 |

> We recommend using larger RAM and VRAM capacities, especially for model training. Also, if you installed the 595 driver, Isaac Sim may crash inside the Docker container later; therefore, we recommend using driver version 580.

### Tool Requirements

| Tool | Version | Notes |
| --- | --- | --- |
| `CUDA` | 12.8 | [Official Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) |
| `Docker` | latest | [Official Guide](https://docs.docker.com/engine/install/ubuntu/) |
| `NVIDIA Container Toolkit` | latest | [Official Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `Hugging Face` | latest | `pip install huggingface-hub`; `huggingface-cli --help` (verify installation) |
| `Git` | latest | `sudo apt update`; `sudo apt install git -y`; `git --version` (verify version) |
| `Miniconda` | latest | [Official Guide](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) (optional) |

## Technical Documentation Index

The complete baseline workflow consists of six stages. **We recommend following the documents in order**; you may also jump directly to the stage you need based on your current progress.

| # | Document | Description |
|---|----------|-------------|
| 1 | [Resource Download](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/1/) | Project overview, key capabilities, hardware requirements, tool requirements, and HuggingFace resource links. |
| 2 | [Environment Setup](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/2/) | Clone the repository, download simulation assets and datasets, build the Docker image, and configure keyboard evdev. |
| 3 | [Quick Start](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/3/) | Start the runtime environment, keyboard teleoperation, data recording, dataset replay, and key mappings. |
| 4 | [Model Training](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/4/) | Training guides for ACT, Diffusion Policy, π₀ (PI0), π₀.₅ (PI05), and SmolVLA with full hyperparameters. |
| 5 | [Policy Inference](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/5/) | Run inference with a trained policy model and automatically record the results. |
| 6 | [4-Camera Real-time Display](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/6/) | Real-time preview from 4 RGB cameras and visualization configuration in teleoperation, recording, replay, and inference modes. |


## Custom Command

Build the docker
```
docker build -t ghrc_2026:v0 .
```

Prepare for the official s2 model:

```
python3 scripts/setup_official_walker_s2.py
```


Start the docker
```
./run.sh 
```
Run the teleop script
```
./teleop_part_sorting.sh
```

### Walker Fixed-Grasp Teleop

Launch the independently developed Walker S2 grasp simulation:

```bash
./teleop_walker_grasp.sh
```

After the scene settles, focus the Isaac Sim window. The keyboard controls are:

| Keys | Action |
|---|---|
| `W/S`, `A/D`, `R/F` | Move the selected hand along X, Y, Z |
| `Y/U`, `V/B`, `N/M` | Rotate the selected hand in roll, pitch, yaw |
| `O` | Switch between the left and right arm |
| `0` | Toggle mirrored bimanual control |
| `K/L` | Open or close the selected hand |
| `+/-` | Change the Cartesian motion step |
| `H` | Return to the ready pose |
| `G` | Run pregrasp, grasp, close, and lift |
| `Q` | Quit |

Arguments for the underlying demo can be passed directly, for example:

```bash
./teleop_walker_grasp.sh --show-hand-colliders --lift-height 0.10
```

The launcher uses the URDF under `assets/resources` by default. Override the
Isaac Sim Python or robot URDF paths when needed:

```bash
ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh \
WALKER_S2_URDF=/path/to/walker_s2.urdf \
./teleop_walker_grasp.sh
```

### Walker S2 IsaacLab Pick/Place RL

This repository also contains a local IsaacLab task for fixed-base Walker S2
pick/place:

```text
Isaac-WalkerS2-PickPlace-IK-v0
```

The task uses a compact 11D palm-IK action:

```text
[palm_xyz_delta, palm_rpy_delta, grip, shoulder_yaw_offset, elbow_yaw_offset, wrist_pitch_offset, wrist_roll_offset]
```

This RL task uses proprioceptive/object/target observations only. It does not
require camera sensors or contact sensor observations.

The current RL workflow uses a behavior-cloning checkpoint as a frozen teacher.
PPO receives the real task reward and an additional decaying BC-prior penalty:

```text
rl_reward = task_reward - bc_coef * ||rl_action - bc_action||^2
```

This lets RL start near the demonstrated arm trajectory while still learning
better contact, grasp timing, and release behavior.

#### 1. Train or refresh the BC teacher checkpoint

The behavior-cloning implementation is included in this repository:

```text
scripts/train_walker_s2_bc.py
scripts/eval_walker_s2_bc.py
```

The repository includes a small fixed BC teacher checkpoint at:

```text
checkpoints/walker_s2_bc/single_demo_phase_processed/best.pt
```

To refresh it from the demo data, run this from the repository root. The
training command writes to `logs/walker_s2_bc/single_demo_phase_processed/best.pt`;
copy it back into `checkpoints/walker_s2_bc/single_demo_phase_processed/best.pt`
if you want to update the packaged teacher.

```bash
cd /home/chris/Projects/internship/zollent_technology/GlobalHumanoidRobotChallenge_2026_Baseline

TERM=xterm /home/chris/IsaacLab/isaaclab.sh -p scripts/train_walker_s2_bc.py \
  --run_name single_demo_phase_processed \
  --demo_roots demos/walker_s2_pick_place_success/walker_s2_pick_place_ep000_20260714_161434.npz \
  --target_key processed_action \
  --append_phase \
  --epochs 800 \
  --batch_size 128 \
  --nonzero_action_weight 5 \
  --grip_action_weight 8 \
  --arm_offset_action_weight 5
```

If the checkpoint already exists, this step can be skipped.

Optionally evaluate the BC teacher directly:

```bash
TERM=xterm /home/chris/IsaacLab/isaaclab.sh -p scripts/eval_walker_s2_bc.py \
  --checkpoint checkpoints/walker_s2_bc/single_demo_phase_processed/best.pt \
  --episodes 1 \
  --settle_steps 180 \
  --print_every 20
```

#### 2. Smoke-test the BC-prior PPO trainer

This small run verifies checkpoint loading, IsaacLab environment creation, and
one PPO update.

```bash
TERM=xterm /home/chris/IsaacLab/isaaclab.sh -p scripts/train_walker_s2_bc_prior_ppo.py \
  --bc_checkpoint checkpoints/walker_s2_bc/single_demo_phase_processed/best.pt \
  --run_name bc_prior_ppo_debug \
  --num_envs 1 \
  --iterations 1 \
  --horizon 8 \
  --epochs 1 \
  --minibatches 1 \
  --settle_steps 0 \
  --rollout_progress_every 1 \
  --headless
```

A successful smoke test reaches lines like:

```text
[BOOT] BC checkpoint loaded before env creation.
[INFO] No settle steps requested. Starting PPO updates.
[ROLLOUT iter=0001 step=0001/8] ...
[ITER 0001] ...
```

#### 3. Run BC-prior PPO training

Start with a moderate training run:

```bash
TERM=xterm /home/chris/IsaacLab/isaaclab.sh -p scripts/train_walker_s2_bc_prior_ppo.py \
  --bc_checkpoint checkpoints/walker_s2_bc/single_demo_phase_processed/best.pt \
  --run_name bc_prior_ppo_less_bc \
  --num_envs 4 \
  --iterations 500 \
  --horizon 128 \
  --epochs 2 \
  --minibatches 2 \
  --settle_steps 0 \
  --bc_coef_start 0.05 \
  --bc_coef_end 0.005 \
  --bc_coef_decay_iters 200 \
  --rollout_progress_every 0 \
  --headless
```

The trainer writes checkpoints and config files to:

```text
logs/walker_s2_bc_prior_ppo/<run_name>/
```

The latest PPO checkpoint is:

```text
logs/walker_s2_bc_prior_ppo/<run_name>/latest.pt
```

#### Training log fields

The PPO trainer prints:

| Field | Meaning |
| --- | --- |
| `reward_mean` | Shaped rollout reward after subtracting the BC-prior penalty. It may be negative early. |
| `recent_ep_return` | Recent raw task episode return. This is the main field to watch for task improvement. |
| `bc_coef` | Current BC-prior penalty coefficient. It decays over training. |
| `teacher_mse` | Mean squared difference between PPO action and BC teacher action. |
| `entropy` | PPO policy exploration level. |

If `teacher_mse` is high and learning stalls, reduce `bc_coef_start` and
`bc_coef_end`. If the arm motion becomes unstable, increase them slightly.

#### Troubleshooting

- If the program appears frozen, use the smoke-test command with
  `--rollout_progress_every 1`. The script prints `[BOOT]`, `[INFO]`,
  `[SETTLE]`, and `[ROLLOUT]` markers around expensive sections.
- The BC checkpoint must be loaded before IsaacLab environment creation. The
  PPO script already does this.
- If IsaacLab raises `No contact sensors added to the prim:
  '/World/envs/env_0/Robot'`, make sure
  `isaaclab_walker_s2/walker_s2_cfg.py` has
  `activate_contact_sensors=False`. This PPO task does not use contact sensor
  observations, and enabling them can fail on some IsaacLab/USD combinations.
- Make sure the robot USD and IK URDF exist after cloning/pulling:
  `assets/resources/walker_s2_description_hand3_v1_left_hand3_v1_right/walker_s2_with_hands_isaaclab.usd`
  and
  `assets/resources/walker_s2_description_hand3_v1_left_hand3_v1_right/walker_s2_description_hand3_v1_left_hand3_v1_right_isaac_simple_hand_collision.urdf`.
- The BC teacher checkpoint must match the env action dim. Current expected
  dimensions are `obs_dim=46` for phase-conditioned BC and `action_dim=11`.
- Run all IsaacLab commands from the repository root so relative checkpoint
  paths resolve correctly.
