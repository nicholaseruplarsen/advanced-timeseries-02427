# Advanced Time Series Analysis (02427)

This repository contains coursework and computational exercises for the Advanced Time Series Analysis course (02427).

## Quick Start

### Prerequisites

Install UV (if you don't have it):
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

Clone the repository:
```zsh
git clone https://github.com/nicholaseruplarsen/advanced-timeseries-02427
```

Change directory:
```zsh
cd advanced-timeseries-02427
```

Open directory with VS Code:
```zsh
code .
```
or Cursor:
```zsh
cursor .
```

Install dependencies:
```zsh
uv sync
```

Then navigate to the assignment directories (e.g., `assignment-2/comp_ex_2_2011.ipynb`).

## Project Structure

```
advanced-timeseries-02427/
├── assignment-1/          # Computer Exercise 1
├── assignment-2/          # Computer Exercise 2  
├── assignment-3/          # Computer Exercise 3  (yet to come)
├── assignment-4/          # Computer Exercise 4  (on its way)
└── src/                   # Shared utilities and models
    ├── models/           # Time series models (SETAR, STAR, IGAR)
    └── utils/            # Estimation and filtering utilities
```

## Dependencies

All dependencies are managed via `pyproject.toml`:
- numpy
- scipy
- matplotlib
- jupyter
- statsmodels
- scikit-learn
- ipykernel

## Notes

- Python 3.11+ has been used, might be (probably not) required
- Plots should be saved to `plots/` directories within each assignment folder.
- The notebooks expect the `src` module to be importable. UV and the `__init.py__` files should handle this automatically

A common issue is getting import errors if you run the notebook cell (maybe also .py file) from another folder than the clone one (advanced-timeseries-02427):
```
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[7], line 1
----> 1 from src.utils.ekg import ekf_parameter_estimation
   ...
ModuleNotFoundError: No module named 'src'
```

In that case, just add the following to the top of your notebook or python file:
```python
import sys
sys.path.append('..')  # or wherever it needs to be
```
