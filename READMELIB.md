<div align="center">

# 🚁 Genesis Drone Env

**High-Fidelity Drone Simulation Environment based on [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)**

[**Documentation**](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hover_env.html) | [**Genesis Engine**](https://github.com/Genesis-Embodied-AI/Genesis)

---

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-demos--usage">Demos</a> •
  <a href="#-hardware-in-the-loop-fpv">FPV Hardware</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

## 📖 Introduction

**Genesis Drone Env** provides a robust playground for drone research, ranging from Reinforcement Learning (RL) to classical Geometric Control. Included in the **official** [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) ecosystem, this repository serves as a foundation for developing complex aerial robotics algorithms.

## 🔖 Related Work (Our Work)

1. [FLARE: Agile Flights for Quadrotor Cable-Suspended Payload Systems via Reinforcement Learning](https://arxiv.org/abs/2508.09797) (Accepted by **IEEE RA-L**) ([Github Code](https://github.com/BEI11HAI/Flare))

### ✨ Features
- **🚀 Reinforcement Learning**: Ready-to-use environments for training tracking policies (PPO included).
- **📐 Geometric Control**: Concise implementation of SO(3)/SE(3) controllers for precise trajectory tracking.
- **🎮 Hardware-in-the-Loop (HIL)**: Connect your real RC transmitter via Flight Controller (FCU) to fly FPV in the simulator.
- **⚙️ Highly Configurable**: Easy tuning of flight parameters, physics settings, and reward functions via YAML.

---

## 💻 Installation

It is recommended to use a virtual environment (conda) to manage dependencies.

```bash
# 1. Create environment
conda create -n genesis_drone python=3.11 # Requires Python >= 3.10
conda activate genesis_drone

# 2. Install Genesis (Ensure you have the latest version)
# Visit https://github.com/Genesis-Embodied-AI/Genesis for detailed instructions

# 3. Clone and install this repository
git clone https://github.com/KafuuChikai/GenesisDroneEnv.git
cd GenesisDroneEnv
pip install -e .
```

---

## 🎬 Demos & Usage

### 1. RL Tracking Task
Train or evaluate a policy to track a moving target.

#### Evaluate Pretrained Model:
```bash
python scripts/eval/track_eval.py
```
<div align="center"> <img src="docs/quick.gif" alt="RL Tracking" width="80%"/> </div>

#### Train Your Own Policy:
```bash
python scripts/train/track_train.py 
```

> **Note:** Training typically converges around the 200th step. You can fine-tune the reward scale in `rl_env.yaml`.

### 2. SE(3) Geometric Controller
Replace the neural network with a classical geometric controller for precise maneuvers.

#### Run Trajectory Tracking:
```bash
python scripts/eval/se3_controller_eval.py --use-trajectory
```

> **Tips:**
> - `--use-trajectory`: Enables circle trajectory tracking.
> - Without the flag, it defaults to Waypoint Mode.

The controller implements the framework proposed in:
- [Geometric tracking control of a quadrotor UAV on SE(3)](https://ieeexplore.ieee.org/document/5717652)
- [Control of Quadrotors Using the Hopf Fibration on SO(3)](https://link.springer.com/chapter/10.1007/978-3-030-28619-4_20)

### 3. 🎮 Hardware-in-the-Loop (FPV)
Fly the simulated drone using your real Radio Controller (RC) via a Flight Controller (FCU) bridge.

<div align="center"> <img src="docs/FPV.gif" alt="FPV Flight" width="80%"/> </div>

**Hardware Setup**

<div align="center"> <img src="./docs/hardware.png" alt="Hardware Connection" width="80%" /> </div>

1. **Prepare FCU**: Use an **STM32H743** FCU.
2. **Flash Firmware**: Flash the custom HEX file: [betaflight_4.4.0_STM32H743_forRC](genesis_drones/utils/BF_firmware_forRC/betaflight_4.4.0_STM32H743_forRC.hex)
3. **Connect**:
   - Power the FCU via USB-C.
   - Connect the FCU's UART port to your PC using a **USB-to-TTL** module.

**Software Configuration**

1. Check your serial port ID (e.g., `/dev/ttyUSB0`) using:
```bash
ls /dev/tty*
```

2. Update the `USB_path` parameter in flight.yaml (or relevant config) to match your port.

**Run FPV Mode**
```bash
python scripts/eval/rc_FPV_eval.py
```

## 🛠 Configuration

Customize the simulation to fit your needs by editing the YAML files in the `config/` directory:

| Config Type | File Path | Description |
| :--- | :--- | :--- |
| **Flight Dynamics** | `config/*/flight.yaml` | PID gains, physical properties (mass, inertia). |
| **Environment** | `config/*/genesis_env.yaml` | Physics engine settings, rendering, scene setup. |
| **RL Hyperparams** | `config/*/rl_env.yaml` | Reward functions, observation space, training steps. |

---

## 🤝 Acknowledgement

This repository is inspired by the following ground-breaking work:

> [**Champion-level drone racing using deep reinforcement learning**](https://www.nature.com/articles/s41586-023-06419-4.pdf) (Nature 2023)

We acknowledge the contributions of the open-source community that make this project possible.