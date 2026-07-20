# Walker S2 Learned Pick-and-Place

This project develops an autonomous one-arm pick-and-place system for the
UBTECH Walker S2 humanoid. The current pipeline uses NVIDIA Isaac Sim and
IsaacLab to build a stable manipulation environment, generate successful
demonstrations with a deterministic staged controller, and train a stage-free
Action Chunking Transformer (ACT) policy to reproduce the task.

The task is to locate the red object, grasp and lift it with the right hand,
carry it to the green target area, lower it, release it, and retreat. The final
student policy runs without the staged controller, teacher phase, scripted
gripper, or task-stage input.

## Current Status

- A registered Walker S2 IsaacLab pick-and-place environment is available.
- The deterministic staged controller completes randomized pick-and-place
  trajectories and generates strict-success demonstrations.
- The recommended learned policy is a state-based, object-relative ACT model.
- The validated ACT checkpoint completed fixed and randomized autonomous
  pick-and-place rollouts in simulation.
- One known environment success predicate can report a timeout for a visually
  valid placement when the object enters the target during the lift dwell.
- Real-robot deployment is not yet connected; a safe state, perception, IK,
  gripper, and watchdog bridge is still required.

## Learning Pipeline

1. Load the Walker S2 robot, table, object, and target into IsaacLab.
2. Execute the validated staged Cartesian controller on randomized object
   poses.
3. Save only complete grasp, lift, carry, release, and retreat trajectories.
4. Train ACT from whole trajectories using recent physical state history and
   future action chunks.
5. Evaluate the checkpoint autonomously on unseen object poses.
6. Prepare a shadow-mode hardware adapter before allowing any real commands.

## Policy Contract

The recommended policy receives an 82-dimensional physical state observation
and predicts a seven-dimensional action:

```text
[object-relative palm position 3, palm orientation offset 3, grip 1]
```

ACT consumes eight recent observations and predicts the next 20 actions. The
environment runs policy decisions at 20 Hz, and overlapping action chunks are
temporally ensembled during evaluation.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `isaaclab_walker_s2/tasks/pick_place/` | IsaacLab scene, observations, actions, rewards, terminations, and Gym registration. |
| `scripts/walker_s2_cartesian_teacher.py` | Deterministic staged pick-and-place teacher. |
| `scripts/generate_walker_s2_cartesian_teacher_demos.py` | Randomized strict-success demonstration generation. |
| `scripts/train_walker_s2_act.py` | Stage-free ACT training. |
| `scripts/walker_s2_act_common.py` | Shared ACT architecture and checkpoint contract. |
| `scripts/eval_walker_s2_act.py` | Autonomous ACT rollout and milestone evaluation. |
| `demos/` | Locally generated demonstrations; not intended for normal Git commits. |
| `logs/` | Locally generated checkpoints and evaluation reports; ignored by Git. |

## Requirements

- Ubuntu with an NVIDIA GPU and compatible driver.
- NVIDIA Isaac Sim and IsaacLab.
- PyTorch from the IsaacLab Python environment.
- Walker S2 simulation assets under `assets/`.

The upstream Walker S2 assets are available from
[UBTECH-Robotics/challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets).
Run IsaacLab-dependent scripts through `isaaclab.sh`; plain system Python does
not provide the complete Isaac Sim runtime.

## Project Setup

Prepare the Walker S2 model and assets:

```bash
python3 scripts/setup_official_walker_s2.py
```

The learned pick-and-place scripts below assume IsaacLab is installed at
`$HOME/IsaacLab`. Use the absolute `isaaclab.sh` path instead if it is installed
elsewhere.

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

#### Stage-free object-relative ACT (recommended student path)

The recommended imitation path uses the registered task:

```text
Isaac-WalkerS2-PickPlace-ObjectRelative-v0
```

Its seven-dimensional action is an absolute palm goal expressed relative to
the current object plus one grip command. The policy receives the complete 82D
state observation, including arm/hand joint state and object/palm/target
geometry. It does not receive teacher stage or phase.

The ACT policy encodes eight recent observations and predicts the next 20
actions. At evaluation, overlapping chunks are temporally ensembled to reduce
single-step drift and transition jitter. The stage controller is used only to
generate strict-success demonstrations and is absent during evaluation.

Collect 64 randomized demonstrations:

```bash
TERM=xterm "$HOME/IsaacLab/isaaclab.sh" -p scripts/generate_walker_s2_cartesian_teacher_demos.py \
  --num_demos 64 \
  --max_attempts 96 \
  --max_steps 300 \
  --randomize_object \
  --seed 3107 \
  --output_dir demos/walker_s2_object_relative_act_teacher64 \
  --headless
```

Train ACT with a whole-trajectory validation split:

```bash
"$HOME/IsaacLab/isaaclab.sh" -p scripts/train_walker_s2_act.py \
  --demo_roots demos/walker_s2_object_relative_act_teacher64 \
  --run_name object_relative_teacher64 \
  --epochs 100
```

Evaluate visibly on unseen randomized object poses:

```bash
"$HOME/IsaacLab/isaaclab.sh" -p scripts/eval_walker_s2_act.py \
  --checkpoint logs/walker_s2_act/object_relative_teacher64/best.pt \
  --episodes 5 \
  --randomize_object \
  --seed 4107 \
  --result_json logs/walker_s2_act/object_relative_teacher64/eval_random5.json
```

The evaluator reports strict ordered grasp, lift, carry, and release
milestones. Offline action error is only a checkpoint-selection signal; unseen
teacher-free success rate is the deployment metric.

##### Current validated result

The validated local checkpoint is:

```text
logs/walker_s2_act/object_relative_teacher64/best.pt
```

It was selected at epoch 93 from 64 strict-success randomized teacher
trajectories. The checkpoint validation metrics were approximately:

```text
first_arm_mae   = 0.018
first_grip_mae  = 0.009
chunk_arm_mae   = 0.016
deployment_score = 0.027
```

A visible fixed-object rollout completed autonomous grasp, lift, carry,
release, and placement without a teacher, phase clock, stage input, or scripted
gripper. In a three-episode randomized pilot, all three rollouts physically
placed and released the object in the green area; two triggered the current
strict environment success termination.

The remaining mismatch is in the success audit, not necessarily in policy
execution. The environment requires five consecutive lifted-and-held steps
while the object is outside the target before it latches the carry history. A
valid randomized trajectory can cross the target boundary during those lift
steps, physically complete the task, and still time out. The standalone
evaluator's milestone audit is slightly less restrictive, so inspect both the
environment termination and the ordered milestone summary when diagnosing a
rollout. Do not retrain solely because of this known false-negative pattern.

Run the validated randomized evaluation with:

```bash
"$HOME/IsaacLab/isaaclab.sh" -p scripts/eval_walker_s2_act.py \
  --checkpoint logs/walker_s2_act/object_relative_teacher64/best.pt \
  --episodes 10 \
  --max_steps 300 \
  --print_every 20 \
  --randomize_object \
  --result_json logs/walker_s2_act/object_relative_teacher64/eval_random10.json
```

The `logs/` directory and `*.pt` files are ignored by Git. A fresh clone must
run the training command above or obtain the checkpoint separately; cloning
this repository alone does not provide `best.pt`.

##### Real-robot deployment status

The checkpoint is validated in IsaacLab but is not ready to be connected
directly to robot motor commands. There is currently no verified real-robot ACT
bridge in this repository.

The policy runs at 20 Hz and consumes an 82D simulator-state observation. A
hardware bridge must reproduce the same feature order, units, and coordinate
frames from real sensors:

- right-arm and right-hand joint positions and velocities;
- right-palm position, orientation, and velocity;
- object position, orientation, and velocity;
- calibrated target position and object/palm/target relative geometry;
- the grasp-state proxies and previous policy action.

The seven-dimensional policy output is not a raw motor command. It contains a
normalized object-relative palm goal and a grip value. Hardware deployment must
decode it with the same workspace, relative-offset, orientation, and quaternion
conventions as `WalkerS2ObjectRelativeCartesianAction`, then use a rate-limited
real-robot IK/servo controller. The simulated gripper scaling must be calibrated
against real hand joint limits rather than copied directly.

Use this deployment order:

1. Run the policy in shadow mode from live robot and perception state without
   transmitting commands.
2. Validate coordinate frames, observation normalization, inference timing,
   and decoded palm targets from the recorded shadow log.
3. Test arm motion with the gripper disabled, reduced workspace and velocity
   limits, a dead-man control, and the robot physically secured.
4. Calibrate the real gripper independently with conservative joint limits.
5. Attempt the complete task with a soft object, an operator at the emergency
   stop, and command-timeout and workspace watchdogs enabled.

Do not deploy this checkpoint to a robot with different kinematics or hand
joint ordering without retraining or an explicitly validated action adapter.

#### Recurrent direct-arm BC (legacy experimental path)

The recurrent BC path trains an autonomous direct-arm policy with this 8D
action layout:

```text
[right_arm_joint_targets_7, grip]
```

It uses complete trajectories rather than shuffled frames. A GRU carries task
history between control steps, while separate heads predict the arm targets,
grip command, and an auxiliary task stage. Teacher phase is used only as an
auxiliary training label; it is not provided to the policy during training or
evaluation. The evaluator does not instantiate the staged controller.

The current feature contract is
`walker_s2_direct_recurrent_state_v2`. Its 46 inputs contain the first 34
physical state observations plus object/palm/target relative positions and
one-step palm/object motion. The eight previous-action observations are
deliberately excluded. This prevents the policy from minimizing offline loss by
copying the teacher's previous action instead of responding to the current
physical state.

The default loader uses these validated demonstration roots:

```text
demos/walker_s2_direct_teacher_random_64
demos/walker_s2_direct_dagger_round2_smooth_pilot
demos/walker_s2_direct_dagger_round2_smooth_batch
```

Train from the repository root:

```bash
TERM=xterm "$HOME/IsaacLab/isaaclab.sh" -p scripts/train_walker_s2_recurrent_bc.py \
  --run_name direct_teacher64_dagger_r2_recurrent_state_v2 \
  --epochs 120
```

The trainer validates every file, splits whole trajectories into training and
validation sets, and writes the accepted/rejected manifest beside the model:

```text
logs/walker_s2_recurrent_bc/direct_teacher64_dagger_r2_recurrent_state_v2/best.pt
logs/walker_s2_recurrent_bc/direct_teacher64_dagger_r2_recurrent_state_v2/dataset_manifest.json
```

The first 100 frames of every trajectory receive additional supervised weight.
This keeps the zero-hidden recurrent startup action accurate instead of hiding
its error inside a much longer trajectory average. Normalized recurrent
features are also clipped consistently during training, evaluation, and DAgger
so unexpected contact cannot turn a small motion delta into an unbounded GRU
input.

First evaluate one visible fixed-object episode:

```bash
TERM=xterm "$HOME/IsaacLab/isaaclab.sh" -p scripts/eval_walker_s2_recurrent_bc.py \
  --checkpoint logs/walker_s2_recurrent_bc/direct_teacher64_dagger_r2_recurrent_state_v2/best.pt \
  --episodes 1 \
  --print_every 50
```

The evaluator refuses legacy action-conditioned features by default, verifies
that the active environment term is the direct-arm action rather than the
staged controller, and reports ordered grasp, lift, carry, and release
milestones. Environment success still requires the strict ordered pick/place
history, a settled object in the target, an open hand, and palm separation.

If the seed policy fails closed-loop, collect a small recurrent DAgger pilot.
The collector runs the recurrent student with persistent GRU state, lets the
existing state-gated teacher intervene smoothly when needed, and records the
teacher action on every student-visited state:

```bash
TERM=xterm "$HOME/IsaacLab/isaaclab.sh" -p scripts/collect_walker_s2_dagger_demos.py \
  --checkpoint logs/walker_s2_recurrent_bc/direct_teacher64_dagger_r2_recurrent_state_v2/best.pt \
  --num_episodes 4 \
  --max_steps 1200 \
  --output_dir demos/walker_s2_recurrent_dagger_state_v2_pilot \
  --save_success_only \
  --intervention_arm_error 0.15 \
  --intervention_grip_error 0.15 \
  --intervention_hold_steps 40 \
  --intervention_blend_in_steps 20 \
  --intervention_blend_out_steps 30 \
  --lift_gate_phase 0.69 \
  --sticky_teacher_after_phase 1.1
```

Add successful pilot trajectories to the next recurrent training round:

```bash
TERM=xterm "$HOME/IsaacLab/isaaclab.sh" -p scripts/train_walker_s2_recurrent_bc.py \
  --demo_roots \
    demos/walker_s2_direct_teacher_random_64 \
    demos/walker_s2_direct_dagger_round2_smooth_pilot \
    demos/walker_s2_direct_dagger_round2_smooth_batch \
    demos/walker_s2_recurrent_dagger_state_v2_pilot \
  --run_name direct_teacher64_dagger_r2_recurrent_state_v2_dagger_r1 \
  --epochs 120
```

Then measure closed-loop generalization over randomized object poses:

```bash
TERM=xterm "$HOME/IsaacLab/isaaclab.sh" -p scripts/eval_walker_s2_recurrent_bc.py \
  --checkpoint logs/walker_s2_recurrent_bc/direct_teacher64_dagger_r2_recurrent_state_v2_dagger_r1/best.pt \
  --episodes 20 \
  --randomize_object \
  --result_json logs/walker_s2_recurrent_bc/direct_teacher64_dagger_r2_recurrent_state_v2_dagger_r1/eval_random_20.json \
  --headless
```

Do not select this policy from frame-level MSE alone. The randomized autonomous
success rate is the deployment metric. The older phase-conditioned BC-prior
workflow below remains available for reproducibility, but its checkpoints are
not compatible with the recurrent evaluator. Legacy recurrent-v1 checkpoints
can still be replayed by the evaluator, but the recurrent DAgger collector
rejects them because their input contract includes the previous action.

The legacy phase-conditioned RL workflow uses a behavior-cloning checkpoint as a frozen teacher.
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
