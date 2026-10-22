"""Feature-set selection and the paper's destination-port frequency feature.

`FeatureSet` sits at the start of every model pipeline, so the port-frequency
table is learned from the rows passed to `fit` (the training partition) only.
Ports never seen in training get frequency 0.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C

FEATURE_SETS = {
    # flow statistics only: no destination-port information at all
    "flow_only": dict(drop_port=True, add_freq=False, port_only=False),
    # every original feature, as used by the paper's baselines
    "with_port": dict(drop_port=False, add_freq=False, port_only=False),
    # the paper's proposed enrichment: original features + port frequency
    "with_port_freq": dict(drop_port=False, add_freq=True, port_only=False),
    # shortcut probe: destination port and its frequency, nothing else
    "port_only": dict(drop_port=False, add_freq=True, port_only=True),
}


class FeatureSet(BaseEstimator, TransformerMixin):
    def __init__(self, name="flow_only"):
        self.name = name

    def fit(self, X, y=None):
        spec = FEATURE_SETS[self.name]
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSet expects a DataFrame with the original column names")
        if spec["add_freq"]:
            counts = X[C.PORT].value_counts(normalize=True)
            self.port_freq_ = {int(k): float(v) for k, v in counts.items()}
        else:
            self.port_freq_ = {}
        if spec["port_only"]:
            base = [C.PORT]
        elif spec["drop_port"]:
            base = [c for c in X.columns if c != C.PORT]
        else:
            base = list(X.columns)
        self.base_columns_ = base
        self.feature_names_out_ = base + (["port_frequency"] if spec["add_freq"] else [])
        return self

    def transform(self, X):
        out = X[self.base_columns_].to_numpy(dtype=np.float32, copy=True)
        if FEATURE_SETS[self.name]["add_freq"]:
            freq = X[C.PORT].map(lambda p: self.port_freq_.get(int(p), 0.0)).to_numpy(np.float32)
            out = np.column_stack([out, freq])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)
