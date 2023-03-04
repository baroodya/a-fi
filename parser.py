import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or predict stock movement with a neural network."
    )
    parser.add_argument("--predict-movement", action="store_true")
    parser.add_argument(
        "-dp",
        "--days-prior",
        metavar="d",
        type=int,
        nargs=1,
        help="the number of days prior a single datapoint includes",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="e",
        type=int,
        nargs=1,
        help="the number of cycles through the data during training",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        metavar="l",
        type=float,
        nargs=1,
        help="the rate wat which the model learns",
    )
    parser.add_argument(
        "-m",
        "--batch-size",
        metavar="m",
        type=int,
        nargs=1,
        help="the number of datapoints in a batch",
    )
    parser.add_argument("--shuffle-dataset", action="store_true")
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument(
        "-p",
        "--val-split",
        metavar="p",
        type=float,
        nargs=1,
        help="the fraction of training data used for validation",
    )
    parser.add_argument(
        "-q",
        "--test-split",
        metavar="q",
        type=float,
        nargs=1,
        help="the fraction of training data used for testing",
    )
    parser.add_argument(
        "-ts",
        "--num-ticker-symbols",
        metavar="t",
        type=int,
        nargs=1,
        help="the number of ticker symbols from which to fetch data",
    )
    parser.set_defaults(
        predict_movement=[False],
        batch_size=[64],
        days_prior=[7],
        epochs=[50],
        learning_rate=[1e-3],
        shuffle_dataset=[False],
        use_pretrained=[False],
        val_split=[0.2],
        test_split=[0.1],
        num_ticker_symbols=[500],
    )
    return parser.parse_args()
