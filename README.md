# Causal Impact Analysis

Measure causal effects of interventions using synthetic control and Bayesian structural time-series.

## Features

- Synthetic control method for counterfactual estimation
- Bayesian structural time-series (BSTS) modeling
- Causal impact quantification with confidence intervals
- Anomaly detection in time-series data
- Pre-intervention trend validation
- Intervention effect decomposition
- Visualization of actual vs counterfactual
- Statistical significance testing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Analyze causal impact:

```bash
python analyze.py --response data.csv --intervention-date 2024-01-01 --output impact.json
```

With control series:

```bash
python analyze.py --response data.csv --control control.csv --intervention-date 2024-01-01
```

Detect anomalies:

```bash
python anomalies.py --data data.csv --sensitivity 2.0 --output anomalies.json
```

Visualize results:

```bash
python visualize.py --impact impact.json --output plot.png
```

## Output

- Point estimate of causal effect
- 95% credible intervals
- Posterior samples for uncertainty quantification
- Pre/post trend comparison
- Statistical test results (Bayesian posterior)

## Example

Before intervention: baseline trend
After intervention: observed vs predicted (counterfactual)
Causal effect: Posterior probability >95% of positive impact
