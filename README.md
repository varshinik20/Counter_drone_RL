# 🛡️ RL-CUAS: Reinforcement Learning for Counter-UAV Swarm Defense

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29+-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

**RL-CUAS** is an autonomous defense simulation and decision-making system designed to protect high-value ground assets from incoming hostile UAV swarms using Deep Reinforcement Learning (DRL). It simulates multi-threat air defense scenarios, trains autonomous commander policies using **Proximal Policy Optimization (PPO)**, benchmarks RL against rule-based heuristic algorithms, and provides interactive visualization interfaces.

---

## 📌 Features

- **Custom 3D Swarm Defense Environment (`SwarmDefenseEnv`)**: Realistic physics and threat dynamics, distance-decaying kill probability ($P_k$), weapon cooldowns, and multiple defense zones (Red/Orange).
- **Deep Reinforcement Learning (PPO)**: Autonomous commander policy trained via `stable-baselines3` to optimize threat engagement and minimize asset damage.
- **Rule-Based Heuristic Baselines**: Classic Threat Assessment and Random policies for statistical benchmarking.
- **Interactive Analytics Dashboard**: Web-based Streamlit dashboard displaying summary KPI cards, comparative distributions, correlation analysis, and training loss/reward curves.
- **3D Isometric Radar Simulator**: Pygame visualizer with side-by-side real-time rendering of **Classic Heuristic vs. DeepRL Agent** with laser beams, hit markers, and video export.
- **Command-Line Interface (`rl-cuas-cli`)**: Unified CLI for training, evaluating, and benchmarking policies.

---

## 📁 Repository Structure

```
counter_uav_project/
├── examples/
│   └── config.yaml             # Scenario and training configuration
├── results/                    # Generated models, plots, metrics, and video recordings
│   ├── models/                 # Saved PPO commander policies
│   ├── figures/                # Visual comparison plots
│   └── summary_metrics.csv     # Policy benchmark summaries
├── src/
│   └── rl_cuas/
│       ├── cli/
│       │   └── main.py         # Command-line interface (rl-cuas-cli)
│       ├── policies/
│       │   └── baselines.py    # Classic heuristic and random policies
│       ├── visualization/
│       │   └── radar_display.py# Pygame isometric simulator & video recorder
│       ├── config.py           # Configuration schema & YAML loader
│       ├── compare.py          # Multi-seed policy benchmark & plotting
│       ├── dashboard.py        # Streamlit analytics dashboard
│       ├── env.py              # Custom Gymnasium swarm defense environment
│       ├── evaluate.py         # Policy evaluation runner
│       ├── train.py            # PPO training pipeline
│       └── training_curves.py  # TensorBoard metric extractor
├── pyproject.toml              # Build metadata & dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/rl-cuas.git
   cd rl-cuas
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

---

## 🚀 Usage & CLI

### 1. Train the PPO Commander Policy
```bash
python -m rl_cuas.cli.main train --config examples/config.yaml
```

### 2. Evaluate a Policy
```bash
python -m rl_cuas.cli.main evaluate --policy deeprl --n_episodes 10 --model_path results/models/commander_policy
```

### 3. Benchmark & Compare Policies
Run multi-seed Monte Carlo simulations across **DeepRL**, **Classic Heuristic**, and **Random** policies:
```bash
python -m rl_cuas.cli.main compare --n_episodes 50 --results_dir results
```

### 4. Launch the Analytics Dashboard
Open the interactive Streamlit dashboard:
```bash
streamlit run src/rl_cuas/dashboard.py
```

### 5. Run the 3D Isometric Radar Visualizer
Run the Pygame simulation to view real-time side-by-side agent performance:
```bash
python -m rl_cuas.visualization.radar_display
```

---

## 📊 Benchmark Summary

| Policy | Damage [%] | Tracking Ratio [%] | Weapon Utilization [%] |
| :--- | :---: | :---: | :---: |
| **DeepRL (PPO)** | Low | High | Optimal |
| **Classic Heuristic** | Medium | Medium | Moderate |
| **Random Baseline** | High | Low | Inefficient |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
