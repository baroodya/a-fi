import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or predict stock movement with a neural network."
    )
    parser.add_argument(
        "-dp",
        "--days-prior",
        metavar="days taken into account for time series",
        type=int,
        nargs="*",
        help="the number of days prior a single datapoint includes",
    )
    parser.add_argument(
        "-s",
        "--sequence-sep",
        metavar="Seperation of prediction time and feature sequence time",
        type=int,
        nargs="*",
        help="the number of days of seperation between the predicted value and the sequence of values used for prediction",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="number of epochs",
        type=int,
        nargs="*",
        help="the number of cycles through the data during training",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        metavar="learning rate",
        type=float,
        nargs="*",
        help="the rate at which the model learns",
    )
    parser.add_argument(
        "-wd",
        "--weight-decay",
        metavar="weight decay",
        type=float,
        nargs="*",
        help="weight decay for L2 regularization",
    )
    parser.add_argument(
        "-m",
        "--batch-size",
        metavar="batch size",
        type=int,
        nargs="*",
        help="the number of datapoints in a batch",
    )
    parser.add_argument("--shuffle-dataset", action="store_true")
    parser.add_argument("--predict-movement", action="store_true")
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument(
        "-p",
        "--val-split",
        metavar="validation split",
        type=float,
        nargs=1,
        help="the fraction of training data used for validation",
    )
    parser.add_argument(
        "-q",
        "--test-split",
        metavar="test split",
        type=float,
        nargs=1,
        help="the fraction of training data used for testing",
    )
    parser.add_argument(
        "-nh",
        "--norm-hist-length",
        metavar="number of days in history used to normalize data",
        type=int,
        nargs="*",
        help="the number of days in history used to normalize data",
    )
    parser.add_argument(
        "-hu",
        "--num-hidden-units",
        metavar="number of hidden units",
        type=int,
        nargs="*",
        help="the number of hidden units in the LSTM model",
    )
    parser.add_argument("--test-best", action="store_true")
    parser.set_defaults(
        batch_size=[64],
        days_prior=[7],
        sequence_sep=[0],
        epochs=[50],
        learning_rate=[1e-3],
        weight_decay=[1e-3],
        shuffle_dataset=False,
        use_pretrained=False,
        val_split=[0.2],
        norm_hist_length=[500],
        num_hidden_units=[16],
        test_best=False,
        predict_movement=False
    )
    return parser.parse_args()
