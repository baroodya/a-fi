# a-fi
## Features
- 📈 Automatic, up to date data collection using `yfinance`
- ⚖️ Rolling Normalization to account for long term growth
- 🤔 Wide LSTM Model that trains quickly, while still maintaining accuracy
- 🛠️ Hyperparameter infrastructure to make fine-tuning simple, easy, and intuitive
- 📲 Availability on Pytorch's `TorchHub` For easy use by others
- ⏳ Variable "Sequence Seperation" for short- and long-term prediction

## Prelimiary Results

![image](https://user-images.githubusercontent.com/59719050/229397473-39065b52-d892-419c-bdcf-005ea0eb5549.png)
This image shows the validation dataset, with targets in blue and predictions in orange. Note the normalized scale on the y-axis.

More than 75% of datapoints in the validation set were within 0.25 of the target. This figure is slightly mysterious, but the average standard deviation used for normalization was around 30. This suggests that the model can predict within around $7 accurately. 

## Future Work

- Exploring deeper, more complex Neural Networks for Increased Accuracy
- Using Sequence Seperation for more conservative investment approaches
- Exploring different investment strategies using the current data
- Export models for multiple different ticker symbols

## Using the Model
To use a pretrained version of this model, use the code below. Currently, this model will only accurately predict NFLX, but support for more ticker symbols is coming soon! Read on to see how to run the code locally and use models pretrained on other ticker symbols or train your own model.
```python
model = torch.hub.load(
    "baroodya/a-fi,
    "ShallowRegressionLSTM",
    force_reload=True,
)
```

### Using the Code
If you want to predict a different ticker symbol, or want to tune hyper parameters, you will have to download and run the code locally. After downloading, make sure you have the required packages:

Using pip: `python3 -m pip -r requirements.txt`

#### Using the Price Prediction Model

From the main directory: `python3 main.py`

To get started with hyper parameters, explore the default settings in [parser.py](https://github.com/baroodya/a-fi/blob/c1c24bba850436a509fe05db9a83f8479b78cc0c/parser.py#L92) and run `python3 main.py -h`.
