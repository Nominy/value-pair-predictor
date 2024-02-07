# value-pair-predictor
Basic PyTorch implementation of LSTM model for value pair trend prediction

It has some data preprocessing scripts included, they target data from Google Finances API, acquired via Google Sheets.
Install dependencies(scikit, pytorch, pandas, numpy, matplotlib)
Be sure to put `data.csv` in `data/` folder, than execute `train.py`.
Model will be saved locally as well as scaler.
Be sure to put `eval_data.csv` in `data/` foler, than execute `eval.py`
Pyplot should appear with price prefiction after succesful model evaluation.
