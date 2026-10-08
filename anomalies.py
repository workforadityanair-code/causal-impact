import argparse
import json
import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, data_series: pd.Series, window_size: int = 20):
        self.data = data_series.values
        self.time_index = data_series.index
        self.window_size = window_size

    def compute_moving_stats(self, window: int) -> Tuple[np.ndarray, np.ndarray]:
        rolling_mean = np.convolve(self.data, np.ones(window)/window, mode='valid')
        rolling_std = np.array([np.std(self.data[i:i+window]) for i in range(len(self.data) - window + 1)])

        padded_mean = np.pad(rolling_mean, (window-1, 0), mode='edge')
        padded_std = np.pad(rolling_std, (window-1, 0), mode='edge')

        return padded_mean, padded_std

    def detect_outliers(self, sensitivity: float = 2.0) -> List[Dict]:
        rolling_mean, rolling_std = self.compute_moving_stats(self.window_size)

        z_scores = np.abs((self.data - rolling_mean) / (rolling_std + 1e-10))
        threshold = sensitivity

        anomalies = []
        for i, z in enumerate(z_scores):
            if z > threshold:
                anomalies.append({
                    'index': int(i),
                    'timestamp': str(self.time_index[i]),
                    'value': float(self.data[i]),
                    'expected': float(rolling_mean[i]),
                    'std': float(rolling_std[i]),
                    'z_score': float(z),
                    'severity': 'high' if z > 3 else 'medium'
                })

        return anomalies

    def detect_trend_change(self, min_segment_length: int = 10) -> List[Dict]:
        changepoints = []

        for i in range(min_segment_length, len(self.data) - min_segment_length):
            before = self.data[:i]
            after = self.data[i:]

            if len(before) >= min_segment_length and len(after) >= min_segment_length:
                stat, p_value = stats.ttest_ind(before, after)

                if p_value < 0.05:
                    changepoints.append({
                        'position': int(i),
                        'timestamp': str(self.time_index[i]),
                        'before_mean': float(np.mean(before)),
                        'after_mean': float(np.mean(after)),
                        'p_value': float(p_value),
                        'confidence': float(1 - p_value)
                    })

        return changepoints

    def detect_seasonality(self, period: int = 7) -> Dict:
        if len(self.data) < 2 * period:
            return {'seasonal': False, 'reason': 'insufficient data'}

        seasonal_components = []
        for i in range(period):
            indices = np.arange(i, len(self.data), period)
            if len(indices) > 1:
                values = self.data[indices]
                seasonal_components.append(np.mean(values))

        cv = np.std(seasonal_components) / (np.mean(seasonal_components) + 1e-10)
        is_seasonal = cv > 0.1

        return {
            'seasonal': bool(is_seasonal),
            'period': period,
            'coefficient_of_variation': float(cv),
            'seasonal_pattern': [float(x) for x in seasonal_components]
        }


def main():
    parser = argparse.ArgumentParser(description='Detect anomalies in time-series')
    parser.add_argument('--data', required=True, help='Time-series CSV file')
    parser.add_argument('--sensitivity', type=float, default=2.0, help='Anomaly sensitivity (z-score threshold)')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    series = df.iloc[:, 0]

    detector = AnomalyDetector(series)

    outliers = detector.detect_outliers(args.sensitivity)
    changepoints = detector.detect_trend_change()
    seasonality = detector.detect_seasonality()

    result = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'anomalies': {
            'outliers': outliers,
            'changepoints': changepoints,
            'seasonality': seasonality
        },
        'summary': {
            'outlier_count': len(outliers),
            'changepoint_count': len(changepoints),
            'has_seasonality': seasonality['seasonal']
        }
    }

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Analysis saved to {args.output}")


if __name__ == '__main__':
    main()
