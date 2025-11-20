#!/bin/bash
# Quick setup for Artifact Registry - run this ONCE before first deployment

set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: No GCP project set"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "Setting up Artifact Registry for project: $PROJECT_ID"
echo ""

# Enable required APIs
echo "1. Enabling required APIs..."
gcloud services enable artifactregistry.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet

# Create Artifact Registry repository
echo ""
echo "2. Creating Artifact Registry repository 'currencycap-repo'..."
gcloud artifacts repositories create currencycap-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="CurrencyCap Portal Docker images" \
  2>/dev/null || echo "Repository already exists (that's OK)"

# Configure Docker auth
echo ""
echo "3. Configuring Docker authentication..."
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# Grant Cloud Build permissions
echo ""
echo "4. Granting Cloud Build permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJECT_ID}@cloudbuild.gserviceaccount.com" \
  --role="roles/run.admin" \
  --condition=None \
  --quiet 2>/dev/null || true

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJECT_ID}@cloudbuild.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" \
  --condition=None \
  --quiet 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "Now you can deploy with:"
echo "  gcloud builds submit --config cloudbuild.yaml"
