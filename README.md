
<h2 align="center">
  <img src='./assets/bench2drive.jpg'>
</h2>

<h2 align="center">
NeurIPS 2024 Datasets and Benchmarks Track
</h2>


<h2 align="center">
  <a href="https://thinklab-sjtu.github.io/Bench2Drive/">Website</a> |
  <a href="https://huggingface.co/datasets/rethinklab/Bench2Drive">Huggingface</a> |
  <a href="https://arxiv.org/abs/2406.03877">arXiv</a> |
  <a href="https://github.com/Thinklab-SJTU/Bench2DriveZoo">Model</a> |
  <a href="https://discord.gg/uZuU3JXVNV">Discord</a>
</h2>

![overview](./assets/overview.jpg)


<h2 align="center">
What can Think2Drive + Bench2Drive provide ? <b>Please click to view the video.</b>
<br>
<b>&#x2193;&#x2193;&#x2193;</b>
</h2>

[![Bench2Drive](https://i.ytimg.com/vi/-osdzJJs2g0/maxresdefault.jpg)](https://www.youtube.com/watch?v=-osdzJJs2g0 "Bench2Drive")

#####
## Table of Contents: <a name="high"></a>
1. [News](#News)
2. [Setup](#setup)
3. [Dataset](#Dataset)
4. [Agents & Models](#agents-models)
5. [Ability Classification](#ability-classification)
6. [Eval Tools](#eval-tools)
7. [Deal with CARLA](#deal-with-carla)
8. [Benchmark](#Benchmark)
9. [License](#license)
10. [Citation](#citation)

## News <a name="news"></a>
  - [2025/02/18] In our latest work [DriveTransformer (ICLR25)](https://openreview.net/forum?id=M42KR4W9P5), **a tiny validation set [Dev10](./leaderboard/data/drivetransformer_bench2drive_dev10.xml) is proposed for quick development of models.** The 10 clips are carefully selected from the official 220 routes, to be both difficult and representative with low variance.  It is suggested to be used for ablation study to avoid overfitting the whole bench2drive220 routes.
  - [2025/02/05] As meticulously described in [CARLA_GARGE](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/common_mistakes_in_benchmarking_ad.md), L2 error is not a meaningful indictor at all. We agree that **authors' should stop reporting results on nuScenes open-loop planning and reviewers should not ask for nuScenes open-loop planning results**. The number on nuScenes open-loop planning is not persuasive at all and could only lead to unjustified and wrong conclusions, which may impede the development of the field.  As a member of the community, we call for stopping using nuScenes open-loop planning as well.
  - [2024/10/14] As kindly pointed out in an [issue](https://github.com/Thinklab-SJTU/Bench2Drive/issues/112), typos were discovered in the ability calculation. We have corrected the typos and updated the multi-ability results. This update will **not affect driving score and success rate**. 
  - [2024/10/14] As kindly pointed out in an [issue](https://github.com/Thinklab-SJTU/Bench2DriveZoo/issues/46), a bug was discovered in the B2D_vad_dataset. We leave it here to be consistent with existing results. We do not anticipate any major influence caused by this bug. Users may fix the bug according to their needs.
  - [2024/09/26] Bench2Drive is accepted at NeurIPS 2024 Datasets and Benchmarks Track.
  - [2024/08/27] We update the latest results under the new protocols with **two new metrics and fixed bugs**. 
  - [2024/08/19] **[Major Updates]** To better assess driving performance, we add two additional metrics: Driving Efficiency and Driving Smoothness. Consequently, we remove the penalty for minimum speed in calculating the Drive Score and extend the TickRunTime from 2000 to 4000 to allow for a more lenient driving evaluation. We are currently reassessing all baselines.
  - [2024/08/10] We update the team_code agent of UniAD and VAD to fix the camera projection bug mentioned in 2024/07/29. Their corresponding scores will be uploaded soon with new metrics introduced.
  - [2024/07/29] As kindly suggested in an [issue](https://github.com/Thinklab-SJTU/Bench2Drive/issues/36), there is a bug in the team code agent of UniAD and VAD during evaluation, i.e, **the BACK CAMERA's extrinsic is wrong**. The training process is correct. To be consistent with the reported results, we do not modify the code right now. Users' are strongly encouraged to use the correct extrinsics.
  - [2024/07/22] We add more reminders in the evaluation code to avoid the miss of logs. According to [Haochen](https://github.com/georgeliu233)'s kind suggestion, we add automatic cleaning code in the evaluation toolkit. Users' may set in their bash script to restart the evaluation infinitely until finishing the evaluation since CARLA is easy to crash.
  - [2024/07/10] We further clean and add more clips in the Full set (13638 clips). Since HuggingFace only allows up to 10000 files per repo, we use two repos to store the Full set. As suggested in this issue [issue](https://github.com/Thinklab-SJTU/Bench2Drive/issues/17), we add a filelist and sha256 of clips for each set.
  - [2024/06/19] Due to a typo in the upload script for HuggingFace, all clips of scenario VehicleTurningRoutePedestrian are empty in the HuggingFace version. We have fixed that. Please check your data to make sure they are not empty. 
  - [2024/06/05] Bench2Drive realases the Full dataset (10000 clips), evaluation tools, baseline code, and benchmark results.
  - [2024/04/27] Bench2Drive releases the Mini (10 clips) and Base (1000 clips) split of the official training data.

## Dataset <a name="dataset"></a>
  - The datasets has 3 subsets, collected by **our strong world model based RL expert [Think2Drive](https://arxiv.org/abs/2402.16720)**, namely Mini (10 clips), Base (1000 clips) and Full (10000 clips), to accommodate different levels of computational resource.
  - [Detailed explanation](docs/anno.md) of dataset structure, annotation information, and visualization of data.

| Subset  | Hugging Face<img src="./assets/hf-logo.png" alt="Hugging Face" width="18"/> | Baidu Cloud<img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Approx. Size | File List|
| :---: |  :---: | :---: | :---: | :---: |
| Mini |   [Download script](https://github.com/Thinklab-SJTU/Bench2Drive/blob/main/tools/download_mini.sh) |  - |  4G | [Mini Json File](./docs/bench2drive_mini_10.json) |
| Base |  [Hugging Face Link](https://huggingface.co/datasets/rethinklab/Bench2Drive) |  [Baidu Cloud Link](https://pan.baidu.com/s/1ZIL-MPhLbgdBYmHkHncn8Q?pwd=1234) |  400G |  [Base Json File](./docs/bench2drive_base_1000.json)|
| Full |  [Full HF Link - 9888 files](https://huggingface.co/datasets/rethinklab/Bench2Drive-Full)/[Sup HF Link - 3814 file](https://huggingface.co/datasets/rethinklab/Bench2Drive-Full-Sup)   | - | 4T | [Full/Sup Json File](./docs/bench2drive_full+sup_13638.json)|

Note that the Mini Set is 10 representative scenes. You may download them by manually select file names from the Base set.

Use the command line: *huggingface-cli download --repo-type dataset --resume-download rethinklab/Bench2Drive --local-dir Bench2Drive-Base* to download from hugginface. User may consider [mirror site](https://hf-mirror.com/) if Huggingface is blocked. Use [BaiduPCS-Go](https://github.com/qjfoidnh/BaiduPCS-Go) to download from Baidu Cloud. Both command lines are resumable.

## Student Model Code (with Think2Drive as Teacher Model)
  - [Uniad/VAD](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad) in Bench2Drive
  - [TCP/ADMLP](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/tcp/admlp) in Bench2Drive

## Setup <a name="setup"></a>

### CARLA Installation
  - Download and setup CARLA 0.9.15
    ```bash
        mkdir carla
        cd carla
        wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
        tar -xvf CARLA_0.9.15.tar.gz
        cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
        cd .. && bash ImportAssets.sh
        export CARLA_ROOT=YOUR_CARLA_PATH
        echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME
    ```

### Model Weights Preparation (for IL Agent)
  - The IL (Imitation Learning) agent requires pre-trained model weights
  - Default weight location: `rl_ppo_model/checkpoints/best_model.pth`
  - Model size: 9.5 MB
  - Training data: Bench2Drive 50GB dataset

### Dataset Preparation
  - Download the appropriate dataset subset (Mini/Base/Full) from the links in the [Dataset](#dataset) section
  - Extract the data to your desired location
  - For IL training, the 50GB RL dataset is recommended: `Bench2Drive-RL50GB/`

## Agents & Models <a name="agents-models"></a>

Bench2Drive supports multiple types of autonomous agents for evaluation. Each agent has different capabilities and use cases.

### 1. Autopilot Agent (Baseline)

**Location**: `leaderboard/team_code/autopilot_agent.py`

**Description**: Uses CARLA's built-in Traffic Manager for autonomous driving. Provides a strong baseline without requiring training data or model weights.

**Key Features**:
  - Track Mode: `Track.MAP` (minimal sensors required)
  - Sensors: GPS + Speedometer only
  - Driving Style: Aggressive (50% faster than speed limit)
  - Traffic Manager Configuration:
    - Respects traffic lights
    - Auto lane change enabled
    - Close following distance (1.5m)

**When to Use**:
  - Quick baseline testing
  - Verifying scenario correctness
  - Comparing against learned models
  - No training data available

**Usage**:
```bash
python3 leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=leaderboard/data/bench2drive220.xml \
    --agent=leaderboard/team_code/autopilot_agent.py \
    --agent-config="" \
    ...
```

### 2. IL Agent (Imitation Learning)

**Location**: `leaderboard/team_code/il_agent.py`

**Description**: Neural network-based agent trained via imitation learning on the Bench2Drive dataset. Uses camera inputs and vehicle states to predict driving actions.

**Key Features**:
  - Track Mode: `Track.SENSORS`
  - Input: RGB camera + vehicle state (6D vector)
  - Output: Throttle, Steering, Brake (3D action)
  - Network Architecture:
    - Image Encoder: 4-layer CNN (32→64→128→256)
    - MLP: 3-layer FC (512→256→3)
    - Dropout: 0.5

**Model Weights**:
  - Primary: `rl_ppo_model/checkpoints/best_model.pth` (9.5 MB)
  - Training: 50 epochs on Bench2Drive-RL50GB dataset
  - Additional checkpoints available: epoch 5, 10, 15, ..., 50

**When to Use**:
  - Learning-based driving behavior
  - Research on imitation learning
  - Evaluating data efficiency
  - Custom driving style training

**Usage**:
```bash
python3 leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=leaderboard/data/bench2drive220.xml \
    --agent=leaderboard/team_code/il_agent.py \
    --agent-config="" \
    ...
```

### 3. RL PPO Agent (Reinforcement Learning)

**Location**: `leaderboard/team_code/rl_ppo_agent.py`

**Description**: Agent based on Proximal Policy Optimization (PPO) algorithm. Designed for reinforcement learning research in autonomous driving.

**Key Features**:
  - Algorithm: PPO (Proximal Policy Optimization)
  - Network Components:
    - Actor Network
    - Critic Network
    - Value Network
  - Configuration: `rl_ppo_model/config/ppo_config.yaml`

**Code Structure**:
```
rl_ppo_model/
├── agents/          # PPO algorithm implementation
├── networks/        # Neural network architectures
├── parameters/      # Parameter management
├── config/          # Configuration files
└── utils.py         # Utility functions
```

**When to Use**:
  - Reinforcement learning research
  - Online learning scenarios
  - Policy optimization experiments
  - Advanced RL algorithm development

**Usage**:
```bash
python3 leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=leaderboard/data/bench2drive220.xml \
    --agent=leaderboard/team_code/rl_ppo_agent.py \
    --agent-config=rl_ppo_model/config/ppo_config.yaml \
    ...
```

### Agent Comparison

| Feature | Autopilot | IL Agent | RL PPO Agent |
|---------|-----------|----------|--------------|
| Implementation | CARLA Traffic Manager | Neural Network | PPO Algorithm |
| Track Mode | MAP | SENSORS | SENSORS |
| Sensors | GPS + Speed | Camera + State | Camera + State |
| Training Required | No | Yes | Yes |
| Training Data | N/A | 50GB Dataset | Environment Interaction |
| Model Weights | N/A | best_model.pth (9.5MB) | PPO checkpoints |
| Driving Style | Aggressive | Learned | Optimized |
| Best For | Baseline | Imitation Learning | RL Research |

### Adding Custom Agents

To add your own agent:

1. Create agent file: `leaderboard/team_code/your_agent.py`
2. Inherit from `AutonomousAgent` base class
3. Implement required methods:
   - `setup(path_to_conf_file)`
   - `sensors()`
   - `run_step(input_data, timestamp)`
   - `destroy()`
4. Link model folder under Bench2Drive directory if needed
5. Run evaluation with your agent

Example structure:
```bash
Bench2Drive/
  leaderboard/
    team_code/
      your_agent.py          # Your agent implementation
  your_model_folder/         # Your model weights/configs
    checkpoints/
    config/
```

## Ability Classification <a name="ability-classification"></a>

Bench2Drive evaluates autonomous driving models across 5 fundamental driving abilities. The 220 test routes are classified into these categories for fine-grained performance analysis.

### The 5 Driving Abilities

#### 1. Overtaking (45 routes)
Tests the ability to safely overtake obstacles and slow-moving vehicles.

**Scenarios** (9 types):
- Accident
- AccidentTwoWays
- ConstructionObstacle
- ConstructionObstacleTwoWays
- HazardAtSideLane
- HazardAtSideLaneTwoWays
- ParkedObstacle
- ParkedObstacleTwoWays
- VehicleOpensDoorTwoWays

#### 2. Merging (80 routes)
Tests the ability to merge into traffic flow and navigate complex junctions.

**Scenarios** (16 types):
- CrossingBicycleFlow
- EnterActorFlow
- HighwayExit
- HighwayCutIn
- InterurbanActorFlow
- InterurbanAdvancedActorFlow
- MergerIntoSlowTraffic
- MergerIntoSlowTrafficV2
- NonSignalizedJunctionLeftTurn
- NonSignalizedJunctionRightTurn
- NonSignalizedJunctionLeftTurnEnterFlow
- ParkingExit
- SequentialLaneChange
- SignalizedJunctionLeftTurn
- SignalizedJunctionRightTurn
- SignalizedJunctionLeftTurnEnterFlow

#### 3. Emergency Brake (60 routes)
Tests the ability to react quickly to unexpected situations and apply emergency braking.

**Scenarios** (12 types):
- BlockedIntersection
- DynamicObjectCrossing
- HardBreakRoute
- OppositeVehicleTakingPriority
- OppositeVehicleRunningRedLight
- ParkingCutIn
- PedestrianCrossing
- ParkingCrossingPedestrian
- StaticCutIn
- VehicleTurningRoute
- VehicleTurningRoutePedestrian
- ControlLoss

#### 4. Give Way (10 routes)
Tests the ability to yield to other vehicles with priority.

**Scenarios** (2 types):
- InvadingTurn
- YieldToEmergencyVehicle

#### 5. Traffic Signs (115 routes)
Tests the ability to recognize and follow traffic rules and signals.

**Note**: This category overlaps with other abilities as it includes 23 scenario types that involve traffic rule compliance.

### Ability-Based Evaluation

The ability classification enables:

**Benefits**:
1. Fine-grained performance analysis
2. Identification of model strengths and weaknesses
3. Targeted model improvement
4. Modular testing (test individual abilities)
5. More stable evaluation (smaller subsets)

**Usage**:
```bash
# Run full evaluation first
python3 leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=leaderboard/data/bench2drive220.xml \
    ...

# Analyze ability-based results
python3 tools/ability_benchmark.py \
    -r your_results.json

# View ability breakdown
cat your_results_ability.json
```

**Ability Definitions**: See `tools/ability_benchmark.py` (lines 16-22) for the complete scenario-to-ability mapping.

## Eval Tools <a name="eval-tools"></a>

### Agent Selection

Bench2Drive provides three evaluation agents:

1. **Autopilot Agent** (`leaderboard/team_code/autopilot_agent.py`)
   - CARLA Traffic Manager baseline
   - No training required
   - Fast and stable

2. **IL Agent** (`leaderboard/team_code/il_agent.py`)
   - Imitation learning based
   - Requires model weights: `rl_ppo_model/checkpoints/best_model.pth`
   - Trained on 50GB Bench2Drive dataset

3. **RL PPO Agent** (`leaderboard/team_code/rl_ppo_agent.py`)
   - Reinforcement learning based
   - Configurable via `rl_ppo_model/config/ppo_config.yaml`
   - For RL research and experiments

### Setup Your Agent
  - Add your agent to leaderboard/team_code/your_agent.py & Link your model folder under the Bench2Drive directory.
    ```bash
        Bench2Drive/
          assets/
          docs/
          leaderboard/
            team_code/
              autopilot_agent.py    # CARLA Autopilot baseline
              il_agent.py           # Imitation Learning agent
              rl_ppo_agent.py       # RL PPO agent
              --> Add your custom agent here
          scenario_runner/
          tools/
          rl_ppo_model/             # Model weights and configs
            checkpoints/
              best_model.pth        # IL agent weights (9.5MB)
            config/
              ppo_config.yaml       # RL PPO configuration
          --> Link your model folder here
    ```

### Evaluation Modes

  - **Debug Mode** - Verify agent correctness on a single route
    ```bash
        # Set GPU_RANK, TEAM_AGENT, TEAM_CONFIG
        bash leaderboard/scripts/run_evaluation_debug.sh
    ```

  - **Multi-Process Multi-GPU Parallel Evaluation** - Fast evaluation on multiple GPUs
    ```bash
        # Set TASK_NUM, GPU_RANK_LIST, TASK_LIST, TEAM_AGENT, TEAM_CONFIG
        # Recommended ratio - GPU:Task = 1:2
        # Example: 4 GPUs can handle 8 parallel tasks
        bash leaderboard/scripts/run_evaluation_multi_uniad.sh
    ```
    Note: Some routes may fail even with multiple retries - this is normal and indicates poor agent performance on those specific scenarios.

### Visualization & Analysis

  - **Create Debug Video** - Generate videos with CAN bus information
    ```bash
        python tools/generate_video.py -f your_rgb_folder/
    ```

  - **Compute Metrics** - Calculate driving score and success rate
    ```bash
        # Merge evaluation JSONs and compute overall metrics
        # Requires exactly 220 routes (failed/crashed routes are acceptable)
        python tools/merge_route_json.py -f your_json_folder/

        # Compute multi-ability performance breakdown
        python tools/ability_benchmark.py -r merge.json

        # Compute driving efficiency and smoothness
        python tools/efficiency_smoothness_benchmark.py -f merge.json -m your_metric_folder/
    ```

    **Important**: The metric calculation assumes 220 total routes. If fewer routes are present, missing ones are scored as 0.

## Deal with CARLA <a name="deal-with-carla"></a>

- CARLA has complex dependencies and is not stable. Please check the issue section of CARLA **very carefully**.
- Use tools/clean_carla.sh frequently and multiple times. Some CARLA processes are difficult to kill. Be sure to clean_carla could avoid lots of bugs.
- In our evaluation tools, the launch of CARLA is automatic: https://github.com/Thinklab-SJTU/Bench2Drive/tree/main/leaderboard/leaderboard/leaderboard_evaluator.py#L203. But you could always start CARLA by the one single command line to debug.
- CARLA is not controlled CUDA_VISIBLE_DEVICES! It is controlled by -graphicsadapter in the command line. **Interestingly, in some machines, for some unknown reasons, -graphicsadapter=1 is not available.** For example, with 4 GPUS, it might be: GPU0 -graphicsadapter=0, GPU1  -graphicsadapter=2, GPU2 -graphicsadapter=3, GPU3  -graphicsadapter=4.
- The conflict of PORT is frequently happened. Use lsof-i:YOUR_PORT frequently to avoid conflict. Avoid use small port numbers (<10000 could be unsafe).
- *4.26.2-0+++UE4+Release-4.26 522 0 Disabling core dumps*. Only showing these two lines without termination is good. *WARNING: lavapipe is not a conformant vulkan implementation, testing use only.* is bad.
- **If you face issues, always try to start CARLA in one single line to make sure CARLA could run.** If CARLA is finished immediately, it is very possible to be related to Vulkan. *Try /usr/bin/vulkaninfo | head -n 5*
- Re-install vulkan might be helpful *sudo apt install vulkan-tools; sudo apt install vulkan-utils*  In the end, you need to make sure your vulkan is correct. We have tested *Vulkan Instance Version: 1.x WARNING: lavapipe is not a conformant vulkan implementation, testing use only.* and version 1.1/1.2/1.3 works fine.
- We find that nvidia driver version 470 is good all the time. 515 has some problems but okay. 550 has lots of bugs.
- *sleep* is important to avoid crash of CARLA. For example, https://github.com/Thinklab-SJTU/Bench2Drive/blob/main/leaderboard/leaderboard/leaderboard_evaluator.py#L207, the sleep time should be extended for slower machines. When it comes to multi-gpu evaluation, https://github.com/Thinklab-SJTU/Bench2Drive/blob/main/leaderboard/scripts/run_evaluation_multi_uniad.sh#L58, the sleep time should also be extended for slower machines.

## Benchmark <a name="benchmark"></a>

- V0.0.3(Currently in use)
  - Fix typos in the ability calculation. ([Issue #112](https://github.com/Thinklab-SJTU/Bench2Drive/issues/112))
  ![benchmark](./assets/benchmark_v3.jpg)

- V0.0.2(Depracated)
  - Fix the BACK CAMERA's extrinsic bug. ([Issue #36](https://github.com/Thinklab-SJTU/Bench2Drive/issues/36))
  - Turn up tickruntime (2000 -> 4000).
  - Drive Score removes min speed penalty.
  - Code Version
    - [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/tree/cd1883c937470ea4d144bf6c7d8922e76d792f07)
    - [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/914b328f26b5534fd13b04d5fc644c8d82196d7b)
  ![depracated benchmark](./assets/benchmark_v2.PNG)

- V0.0.1(Depracated)
  - Initial Version
  - Code Version
    - [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/tree/ee62902eeb8a9f7ab50fa62009c2de12bf645220)
    - [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/31432e868c3ca1bef5c7fa39ba4bd4e7a3e7538a)
  ![depracated benchmark](./assets/benchmark.jpg)

## License <a name="license"></a>

All assets and code are under the [CC-BY-NC-ND](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>

Please consider citing our papers if the project helps your research with the following BibTex:

```bibtex
@inproceedings{jia2024bench,
  title={Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author={Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  booktitle={NeurIPS 2024 Datasets and Benchmarks Track},
  year={2024}
}

@inproceedings{li2024think,
  title={Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving (in CARLA-v2)},
  author={Qifeng Li and Xiaosong Jia and Shaobo Wang and Junchi Yan},
  booktitle={ECCV},
  year={2024}
}
```

