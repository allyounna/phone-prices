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

##### Data Managment

- Install dvc

```
pip install dvc
```

- Add [data](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

```
dvc add ./data/train.csv
dvc add ./data/test.csv
```

- Create dvc storage

```
mkdir ../dvc_storage
dvc remote add -d localremote ../dvc_storage
dvc push
```
