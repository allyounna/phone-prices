from mlflow.tracking import MlflowClient

client = MlflowClient()

# Specify model name and version you want to promote
model_name = "PriceModel"
version_to_promote = "2"

# Promote to Production stage
client.transition_model_version_stage(
    name=model_name,
    version=version_to_promote,
    stage="Production",
    archive_existing_versions=True,  # Optional, archives any existing Production
)

print(f"Version {version_to_promote} of {model_name} is now Production")
