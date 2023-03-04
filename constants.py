MOVEMENT_MODEL_PATH = (
    "./movement_prediction/models/"
)

PRICE_MODEL_PATH = (
    "./price_prediction/models/"
)

TRAINING_WEIGHTS_FILE_NAME = "best_training_model_weights.pth"
VAL_WEIGHTS_FILE_NAME = "best_val_model_weights.pth"
TEST_WEIGHTS_FILE_NAME = "best_test_model_weights.pth"
STATS_FILE_NAME = "best_model_stats.txt"

TRAIN_TICKER_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "NVDA",
    "TSLA",
    "META",
    "AVGO",
    "ORCL",
    "CSCO",
    "ADBE",
    "TXN",
    "NFLX",
    "CRM",
    "QCOM",
]

TEST_TICKER_SYMBOLS = ["IBM"]

NUM_HISTORICAL_DAYS = 7

SINGLE_TICKER_SYMBOL = ["AAPL"]
