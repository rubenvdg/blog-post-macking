This repository contains the code to reproduce the results from the blog post [A New Method for Multi-Horizon Forecasting with a Single Tree-based Model](url).

1. [Download the data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
1. Add `calendar.csv` and `sales_train_validation.csv` to `data/`.
1. Make a virtual environment with Python 3.11.
1. Install the requirements with `pip install -r requirements.txt`.
1. Run `feature-engineering.ipynb` to generate `data/features.parquet`.
1. Run `cross-validation.ipynb` to run the cross validation.
