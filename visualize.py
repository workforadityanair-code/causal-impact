import argparse
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultVisualizer:
    def __init__(self, response_data: str, impact_json: str):
        self.df = pd.read_csv(response_data)
        self.response = self.df.iloc[:, 0].values

        with open(impact_json, 'r') as f:
            self.impact = json.load(f)

    def plot_impact(self, output_file: str = None, figsize: Tuple = (12, 6)):
        pre_idx = self.impact['pre_period_length']
        counterfactual = np.zeros_like(self.response)

        pre_mean = self.response[:pre_idx].mean()
        counterfactual[:] = pre_mean

        post_effect = self.impact['impact']['point_effect']
        counterfactual[pre_idx:] = pre_mean + post_effect

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        time_index = np.arange(len(self.response))

        ax1.plot(time_index, self.response, 'b-', linewidth=2, label='Actual')
        ax1.plot(time_index, counterfactual, 'r--', linewidth=2, label='Counterfactual')
        ax1.axvline(x=pre_idx, color='gray', linestyle=':', linewidth=1)
        ax1.fill_between(time_index[pre_idx:], counterfactual[pre_idx:], alpha=0.3, color='yellow', label='Causal effect')
        ax1.set_ylabel('Response')
        ax1.set_title('Causal Impact Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        effect = self.response[pre_idx:] - counterfactual[pre_idx:]
        ax2.bar(time_index[pre_idx:], effect, color=['green' if x > 0 else 'red' for x in effect], alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Effect')
        ax2.set_title('Pointwise Causal Effect')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150)
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()

    def generate_summary(self):
        impact = self.impact['impact']

        summary = f"""
Causal Impact Analysis Summary
{"="*50}
Intervention Date: {self.impact['intervention_date']}
Pre-period length: {self.impact['pre_period_length']}
Post-period length: {self.impact['post_period_length']}

Point Effect: {impact['point_effect']:.2f}
Cumulative Effect: {impact['cumulative_effect']:.2f}
95% CI: [{impact['ci_lower']:.2f}, {impact['ci_upper']:.2f}]
Percent Change: {impact['percent_change']:.1f}%

p-value: {impact['p_value']:.4f}
Significant: {impact['significant']}

Interpretation: {self.impact['interpretation']}
"""
        return summary


def main():
    parser = argparse.ArgumentParser(description='Visualize causal impact results')
    parser.add_argument('--response', required=True, help='Response CSV')
    parser.add_argument('--impact', required=True, help='Impact JSON from analyze.py')
    parser.add_argument('--output', help='Output plot file')

    args = parser.parse_args()

    visualizer = ResultVisualizer(args.response, args.impact)

    print(visualizer.generate_summary())

    visualizer.plot_impact(args.output)


if __name__ == '__main__':
    main()
