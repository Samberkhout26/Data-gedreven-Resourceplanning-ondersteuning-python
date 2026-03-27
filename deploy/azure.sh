#!/bin/bash
# Azure deployment: Container Registry + Container App voor Rister ML Pipeline
#
# Vereisten:
#   - Azure CLI geïnstalleerd (az login gedaan)
#   - Prefect Cloud account + API key
#   - Docker Desktop draaiende
#
# Gebruik:
#   bash deploy/azure.sh
#
# Pas de variabelen hieronder aan naar jouw Azure omgeving.

set -e  # stop bij fout

# ─── Configuratie ──────────────────────────────────────────────────────────────
RESOURCE_GROUP="rister-ml"
LOCATION="westeurope"
ACR_NAME="ristermlregistry"          # moet uniek zijn in Azure, alleen kleine letters
CONTAINER_APP_ENV="rister-env"
CONTAINER_APP_NAME="rister-worker"
IMAGE_NAME="rister-pipeline"
IMAGE_TAG="latest"

# Prefect Cloud (haal op via: prefect cloud workspace ls)
PREFECT_API_URL="https://api.prefect.cloud/api/accounts/<account-id>/workspaces/<workspace-id>"
PREFECT_API_KEY="pnu_xxxxxxxxxxxxxxxxxxxx"

# Database (Azure PostgreSQL)
POSTGRES_URI="postgresql://gebruiker:wachtwoord@jouw-server.postgres.database.azure.com:5432/rister_prod17"

# MLflow (lokaal of Azure ML)
MLFLOW_URI="http://127.0.0.1:5001"
# ───────────────────────────────────────────────────────────────────────────────

echo "=== 1. Resource group aanmaken ==="
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "=== 2. Container Registry aanmaken ==="
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true

echo "=== 3. Docker image bouwen en pushen ==="
az acr build \
  --registry "$ACR_NAME" \
  --image "$IMAGE_NAME:$IMAGE_TAG" \
  .

echo "=== 4. Container Apps omgeving aanmaken ==="
az containerapp env create \
  --name "$CONTAINER_APP_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "=== 5. Container App aanmaken (Prefect worker) ==="
ACR_SERVER="${ACR_NAME}.azurecr.io"
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

az containerapp create \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINER_APP_ENV" \
  --image "${ACR_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_NAME" \
  --registry-password "$ACR_PASSWORD" \
  --cpu 2 \
  --memory 4Gi \
  --min-replicas 1 \
  --max-replicas 1 \
  --env-vars \
    "PREFECT_API_URL=${PREFECT_API_URL}" \
    "PREFECT_API_KEY=${PREFECT_API_KEY}" \
    "POSTGRES_URI=${POSTGRES_URI}" \
    "MLFLOW_URI=${MLFLOW_URI}" \
  --command "prefect" "worker" "start" "--pool" "rister-pool"

echo ""
echo "=== Deployment klaar ==="
echo "Worker draait in: https://portal.azure.com/#resource/resourceGroups/${RESOURCE_GROUP}"
echo "Prefect UI:       https://app.prefect.cloud"
echo ""
echo "Volgende stap: maak de deployment aan via:"
echo "  python src/flows/deploy.py"
