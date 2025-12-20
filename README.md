<div align="center">
  <h1>phone-price</h1>
  <em>An ML-powered tool that accurately predicts smartphone market prices by analyzing key features.</em>
  <br />
  <br />
  <p align="center">
    <a href="#"/>
      <img src="https://img.shields.io/badge/python-3.12-blue">
    </a>
    <a href="https://github.com/astral-sh/uv"/>
      <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json">
    </a>
    <a href="https://github.com/astral-sh/ruff"/>
      <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
    </a>
    <a href="https://microsoft.github.io/pyright"/>
      <img src="https://microsoft.github.io/pyright/img/pyright_badge.svg">
    </a>
    <a href="https://github.com/pre-commit/pre-commit"/>
      <img src="https://img.shields.io/badge/pre--commit-enabled-blue?logo=pre-commit">
    </a>
    <a href="https://conventionalcommits.org"/>
      <img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white&color=blue">
    </a>
  </p>
</div>

##### Development

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and sync environment:

```
uv sync --frozen
```

- Activate project environment:

```
source .venv/bin/activate
```

- Don't forget to use `pre-commit`:

```
pre-commit install
```

#### Data managment

Get env.sh from your colleague

```
source env.sh
```

then you can pull dataset and other essentials

```
dvc pull
```

### Run mlflow server with

```
bash src/services/run_mlflow.sh
```

### Run mlflow inference server

Create docker for your model

```
mlflow models build-docker -m models:/PriceModel/1 -n price-model:latest
```

and run it!

```
docker run -p 6000:8080 price-model:latest
```

or

```
mlflow models serve -m models:/PriceModel/1 --no-conda -p 6000
```

if you want complete docker

test your container with

```
curl -X POST http://localhost:6000/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "data": [[4096, 5000, 6.5, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        }
      }'
```

### Run training with

```
python src/train.py
```

### Run inference with

```
python src/predict.py --input_data data/test.csv
```

### Config params:

```
data:
  test_size: percentage of test data (float)
  random_state: random state for reproducibility (int)

model:
  name: name of chosen model ("random_forest", "svm", "logistic_regression")
  save: the flag for saving the model (1 - saving, 0 - not saving)
  params:
    random_forest:
      n_estimators: amount of trees (int)
      max_depth: maximal depth of trees (int)
    svm:
      kernel: the type of core for data conversion ("linear", "poly", "rbf", "sigmoid")
      C: regularization parameter (float)
    logistic_regression:
      max_iter: maximum number of training iterations (int)
      C: regularization parameter (float)
```
