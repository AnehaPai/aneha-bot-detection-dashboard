"""Imbalance-aware Bot detection on CIC-IDS2017 Friday-morning traffic."""
import os

# joblib/loky cannot query physical cores on this Windows setup and warns on every
# parallel fit; pin it to the same conservative value used everywhere else.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
