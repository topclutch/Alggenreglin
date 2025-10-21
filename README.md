# 🧬 Genetic Algorithm for Data Mining / Regression

This project implements a **Genetic Algorithm (GA)** to solve **data mining problems**, such as:  
- Feature selection  
- Linear and non-linear regression model fitting  
- Hyperparameter optimization  

It includes scripts to prepare data, execute the genetic algorithm, and evaluate the results.

---

## 📂 Contents

- `README.md` — this file  
- `requirements.txt` — dependencies (if available)  
- `data/` — folder containing example datasets (CSV)  
- `src/` — source code (genetic algorithm, evaluation, utilities)  
- `notebooks/` — Jupyter notebooks for experimentation and visualization  
- `scripts/` — executable scripts (e.g., `run_experiment.py`, `train.py`)  
- `results/` — output of runs (models, metrics, charts)  

> Adjust the structure above according to your actual folders and filenames.

---

## ⚙️ Features

- **Modular implementation** of a Genetic Algorithm (GA)  
- Representation of **individuals (chromosomes)** for regression and/or feature selection problems  
- Genetic operators: **selection**, **crossover**, **mutation**, and **replacement**  
- Evaluation using metrics such as `MSE`, `RMSE`, and `R²`  
- Support for experimentation with different parameters: population size, number of generations, mutation rate, etc.

---

## 🧩 Requirements

- Python **3.8+** (recommended)
- Main libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`  

Example of a minimal `requirements.txt`:

```txt
numpy
pandas
scikit-learn
matplotlib
tqdm
