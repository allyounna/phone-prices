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
phone-prices-train
```

to change the model settings use:

```
phone-prices-train model=svm
```

to see other options put:

```
phone-prices-train --help
```

### Run inference with

```
phone-prices-predict
```
