dependencies = ['torch']
from architectures import ShallowRegressionLSTM

# resnet18 is the name of entrypoint
def resnet18(num_features=5, hidden_units=32, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = ShallowRegressionLSTM(num_features=num_features, hidden_units=hidden_units, **kwargs)
    return model