Project: Tetris DDQN/DQN training (workspace)

This workspace contains multiple variants of a Tetris environment and several training scripts for DQN / Double DQN agents. The repository appears to be a student project / coursework with several parallel copies (in `new/`, `colley_25512144/`) and multiple agent/training implementations.

This README explains the layout, how to set up a Python environment on Windows (PowerShell), how to run the training scripts that are present, and lists fixes and compatibility notes made while debugging.

---

Contents (important files and folders)

- `agent.py`                                    - alternative agent implementation
- `env_tetris.py`                               - modified environment (Gym-like); reset/step return observation(s)
- `train.py`                                    - top-level training script (different agent)
- `new/` folder                                  - an alternate working copy with `env_tetris_.py`, `trainingddqn.py`, etc.
- `colley_25512144/` folder                      - another copy with `colley_25512144_ddqn(AGENT).py`, `colley_25512144_ddqn(TRAIN).py`, etc.

Files you will likely run

- `new/training.py`                             - training driver which uses `env_tetris_` (in same folder)
- `colley_25512144/colley_25512144_ddqn(TRAIN).py` - training driver referencing `env_tetris` and local agent
- `train.py`                                    - other top-level driver (check comments in file)

Note: There are several copies/variants of agent and training code. Review the header comments of the script you intend to run and pick the matching agent/environment in the same folder.

---
# Tetris DDQN/DQN — quick start

This workspace contains multiple variants of a Tetris environment and training scripts for DQN / Double-DQN agents.

The instructions below are intentionally short: install the required packages, then run the training script you want.

## Requirements

- Python 3.8+ (3.11 tested here)
- pip

Required Python packages (minimum):

- numpy
- matplotlib
- pygame
- tensorflow

## Quick setup (PowerShell)

1) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install packages:

```powershell
python -m pip install --upgrade pip
pip install numpy matplotlib pygame tensorflow
```

## Run the training scripts

From the `dqn` folder (workspace root for the project), run one of the scripts below. Run from the workspace root so imports resolve correctly.

Example 1 — the `new` copy (uses `env_tetris_` in the same folder):

```powershell
python -u .\new\training.py
```

Example 2 — the `colley_25512144` copy (uses local agent file):

```powershell
python -u .\colley_25512144\colley_25512144_ddqn(TRAIN).py
```

## Notes

- If a `ModuleNotFoundError` appears (e.g. for `env_tetris` or `ddqn_agent`), make sure you are running from the workspace root and that your virtual environment is activated.
- If you see TensorFlow CPU-optimization warnings (about SSE/AVX), they are informational only — TensorFlow will still run.
- To suppress game windows while training, instantiate the environment with `TetrisApp(render=False)` in the script.

## Need anything else?

If you want, I can:
- Create a `requirements.txt` with pinned versions.
- Run a quick import test for a specific script to confirm imports build without errors.
- Normalize and rename files across the workspace to remove non-standard filenames.
