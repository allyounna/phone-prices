import argparse

from mlflow import MlflowClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile Price Classification Training")

    parser.add_argument(
        "--name",
        type=str,
        help="Name of the registered model to create",
    )

    args = parser.parse_args()
    client = MlflowClient()
    client.create_registered_model(args.name)
