"""Fixed paths, seeds and protocol constants shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Friday-WorkingHours-Morning.pcap_ISCX.csv"
RAW_SHA256 = "53a41c24d570ea83b7ac55b2e94df94e7a8216aeb80a2af0246b6bc8bb543000"

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

LABEL = "Label"
PORT = "Destination Port"
POSITIVE = "Bot"
NEGATIVE = "BENIGN"
DUPLICATE_COLUMN = "Fwd Header Length.1"  # exact copy of "Fwd Header Length" (verified in audit)

SEED = 42
TEST_SIZE = 0.33        # paper: 67/33 stratified train_test_split
VAL_SIZE = 0.20         # carved out of the training partition, used only for model selection
ROBUSTNESS_SEEDS = [0, 1, 2, 3, 4]

N_JOBS = 2              # conservative parallelism: the machine is memory-constrained
