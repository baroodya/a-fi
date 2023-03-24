dependencies = ['torch']
from architectures import ShallowRegressionLSTM as _ShallowRegressionLSTM

# resnet18 is the name of entrypoint
def ShallowRegressionLSTM(sequence_length=5, num_features=4, hidden_units=32, **kwargs):
    """ # This docstring shows up in hub.help()
    ShallowLSTM
    num_features: The number of features in the input vector
    hidden_units: the size of the hidden state in the LSTM
    """
    # Call the model, load pretrained weights
    model = _ShallowRegressionLSTM(sequence_length=sequence_length, num_features=num_features, hidden_units=hidden_units, **kwargs)
    return model