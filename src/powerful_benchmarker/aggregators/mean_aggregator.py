from .base_aggregator import BaseAggregator
import numpy as np

class MeanAggregator(BaseAggregator):
    def get_aggregate_performance(self, accuracy_per_split):
        return np.mean(list(accuracy_per_split.values()))



    