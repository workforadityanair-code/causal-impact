import argparse
import json
import logging
from typing import Tuple, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalImpactAnalyzer:
    def __init__(self, response_series: pd.Series, control_series: pd.Series = None):
        self.response = response_series.values
        self.control = control_series.values if control_series is not None else None
        self.time_index = response_series.index

    def split_at_date(self, intervention_date: str) -> Tuple[int, int]:
        dates = pd.to_datetime(self.time_index)
        intervention_date = pd.to_datetime(intervention_date)
        split_idx = (dates <= intervention_date).sum()
        return split_idx, len(dates)

    def fit_synthetic_control(self, pre_period_idx: int) -> Tuple[np.ndarray, float]:
        response_pre = self.response[:pre_period_idx]

        if self.control is None:
            counterfactual = response_pre.mean() + np.zeros_like(self.response)
            r_squared = 0.0
        else:
            control_pre = self.control[:pre_period_idx]

            model = LinearRegression()
            model.fit(control_pre.reshape(-1, 1), response_pre)
            counterfactual = model.predict(self.control.reshape(-1, 1))

            y_pred = model.predict(control_pre.reshape(-1, 1))
            ss_res = np.sum((response_pre - y_pred) ** 2)
            ss_tot = np.sum((response_pre - response_pre.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return counterfactual, r_squared

    def compute_impact(self, counterfactual: np.ndarray, pre_period_idx: int) -> Dict[str, Any]:
        response_post = self.response[pre_period_idx:]
        counterfactual_post = counterfactual[pre_period_idx:]

        point_effect = response_post - counterfactual_post
        cumulative_effect = np.sum(point_effect)

        response_pre = self.response[:pre_period_idx]
        response_std = np.std(response_pre)

        se = response_std / np.sqrt(len(response_post))

        z_score = cumulative_effect / (se * np.sqrt(len(response_post)) + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        ci_lower = cumulative_effect - 1.96 * se * np.sqrt(len(response_post))
        ci_upper = cumulative_effect + 1.96 * se * np.sqrt(len(response_post))

        avg_effect = np.mean(point_effect)
        pct_change = (avg_effect / np.mean(counterfactual_post)) * 100 if np.mean(counterfactual_post) != 0 else 0

        return {
            'point_effect': float(avg_effect),
            'cumulative_effect': float(cumulative_effect),
            'effect_std': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'p_value': float(p_value),
            'percent_change': float(pct_change),
            'significant': p_value < 0.05
        }

    def analyze(self, intervention_date: str) -> Dict[str, Any]:
        pre_idx, total_idx = self.split_at_date(intervention_date)

        logger.info(f"Pre-period: {pre_idx} observations, Post-period: {total_idx - pre_idx} observations")

        counterfactual, r_squared = self.fit_synthetic_control(pre_idx)
        impact = self.compute_impact(counterfactual, pre_idx)

        result = {
            'intervention_date': intervention_date,
            'timestamp': datetime.utcnow().isoformat(),
            'pre_period_length': pre_idx,
            'post_period_length': total_idx - pre_idx,
            'model_fit_r_squared': float(r_squared),
            'impact': impact,
            'interpretation': self._interpret_impact(impact)
        }

        return result

    def _interpret_impact(self, impact: Dict[str, Any]) -> str:
        if not impact['significant']:
            return 'No statistically significant causal effect detected'

        if impact['point_effect'] > 0:
            return f"Positive causal effect: +{impact['percent_change']:.1f}%"
        else:
            return f"Negative causal effect: {impact['percent_change']:.1f}%"


def main():
    parser = argparse.ArgumentParser(description='Causal impact analysis')
    parser.add_argument('--response', required=True, help='Response time-series CSV')
    parser.add_argument('--control', help='Control time-series CSV')
    parser.add_argument('--intervention-date', required=True, help='Intervention date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    df_response = pd.read_csv(args.response)
    response_series = df_response.iloc[:, 0]

    control_series = None
    if args.control:
        df_control = pd.read_csv(args.control)
        control_series = df_control.iloc[:, 0]

    analyzer = CausalImpactAnalyzer(response_series, control_series)
    result = analyzer.analyze(args.intervention_date)

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Analysis saved to {args.output}")


if __name__ == '__main__':
    main()
