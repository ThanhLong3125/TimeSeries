"""
Lab 4: Multivariate Time Series Forecasting (Electricity Load + Weather)
Modules:
- logger: Centralized logging setup
- tracker: Experiment audit trail & MD5 hashing
- data_loader: Loading, validation, and StandardScaler normalization
- feature_engineer: 24h lag windows, rolling features, 3D tensor preparation
- models_var: Vector Autoregression statistical model with ADF test
- models_xgb: XGBoost Machine Learning model with early stopping
- models_lstm: Deep Learning PyTorch LSTM model (LSTM 64 -> 32 -> Dense 1)
- evaluator: RMSE, MAE, MAPE evaluation and backtesting
- visualizer: Academic plots (EDA, benchmarks, loss curves, forecasts)
"""
