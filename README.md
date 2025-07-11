# cage-challenge-4

This repository contains three independent reinforcement learning implementations: **IPPO**, **MAPPO**, and **HMARL**. Each directory includes its own training script and can be run independently.

---

## 📁 Project Structure

```
cage-challenge-4/
│
├── Ippo/
│   └── ippo_hyperparameter.py
│
├── Mappo/
│   └── mappo_hyperparameter.py
│
└── Hmarl/
    └── train_subpolicies.py
```

---

## ✅ Setup & Requirements

It’s recommended to use a **virtual environment** to manage dependencies.

### 1️⃣ Create and activate a virtual environment

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install requirements

Make sure you have a `requirements.txt` file in the project root.  
Install all dependencies with:

```bash
pip install -r requirements.txt
```


## 🚀 How to Run

### ⚡ IPPO

Navigate into the `Ippo` directory and run:

```bash
python ippo_hyperparameter.py [options]
```

**Available arguments for IPPO:**

| Argument | Description | Example |
| -------- | ----------- | ------- |
| `--cluster` | Run under SLURM (submit job via `sbatch`). | `--cluster` |
| `--no-optuna` | Disable Optuna hyperparameter tuning and use fixed hyperparameters. | `--no-optuna` |
| `--lr` | Learning rate (float). | `--lr 5e-5` |
| `--clip-param` | PPO clipping ratio. | `--clip-param 0.2` |
| `--train-batch-size` | Total training batch size. | `--train-batch-size 150000` |
| `--minibatch-size` | Minibatch size. | `--minibatch-size 4096` |

**Example:**

```bash
cd Ippo
python ippo_hyperparameter.py --no-optuna --lr 5e-5 --clip-param 0.2 --train-batch-size 150000 --minibatch-size 4096
```

---

### ⚡ MAPPO

Navigate into the `Mappo` directory and run:

```bash
cd Mappo
python mappo_hyperparameter.py
```

_No additional arguments are required for MAPPO._

---

### ⚡ HMARL

Navigate into the `Hmarl` directory and run:

```bash
cd Hmarl
python train_subpolicies.py
```

_No additional arguments are required for HMARL._

---

## ✅ Requirements

Make sure to install the required Python packages.  
You can create a `requirements.txt` and install with:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- The `--cluster` flag is useful if running on a compute cluster with SLURM.
- By default, IPPO uses Optuna for hyperparameter tuning. Use `--no-optuna` to disable tuning and use your fixed hyperparameters.

---


