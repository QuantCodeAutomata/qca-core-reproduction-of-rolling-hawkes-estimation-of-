# Rolling Hawkes Estimation of Market Endogeneity

Reproduction of the core empirical results from the paper on Hawkes-process-based
measurement of short-term market endogeneity using E-mini S&P 500 futures (ES) data.

## Overview

This repository implements:

- **exp_1**: Core rolling Hawkes estimation of market endogeneity from ES mid-price changes (1998–2010)
- **exp_2**: Synthetic validation of within-second timestamp-rounding bootstrap
- **exp_3**: Reshuffled-timing control to test whether n is more than an activity proxy
- **exp_4**: Flash-crash case study (April 27 vs May 6, 2010)

## Repository Structure

```
repo/
├── data/               # Data loading and storage (Parquet caches)
├── src/
│   ├── models/         # Hawkes MLE, likelihood, kernel
│   ├── data/           # Data loading, order-book reconstruction, day filtering
│   ├── diagnostics/    # Residual transformation, KS tests, rejection logic
│   ├── rolling/        # Rolling-window pipeline
│   ├── sim/            # Hawkes simulation (Ogata thinning)
│   ├── validation/     # Timestamp-rounding bootstrap validation
│   ├── control/        # Reshuffled-timing control
│   └── case_study/     # Flash-crash case study
├── exp/                # Experiment entry-point scripts
├── tests/              # pytest unit and integration tests
├── results/            # Outputs: figures, tables, RESULTS.md
├── AGENTS.md
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Set your Massive API key:
```bash
export MASSIVE_TOKEN=your_api_key_here
```

## Running Experiments

```bash
# Experiment 1 – Core rolling Hawkes estimation
python exp/run_exp1.py

# Experiment 2 – Synthetic timestamp-rounding validation
python exp/run_exp2.py

# Experiment 3 – Reshuffled timing control
python exp/run_exp3.py

# Experiment 4 – Flash-crash case study
python exp/run_exp4.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for a summary of all experiment findings.
