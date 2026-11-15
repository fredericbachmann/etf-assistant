#!/bin/bash
# Deploy the checked-out commit: build and push the images, then update the server.
# Used by .github/workflows/deploy.yml and by hand (AWS_PROFILE=... deploy/deploy.sh).
# Needs docker, the aws CLI with credentials, and the etf-assistant CloudFormation stack.
set -euo pipefail
cd "$(dirname "$0")/.."

STACK=etf-assistant
output() { aws cloudformation describe-stacks --stack-name $STACK --output text \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue"; }
REGISTRY=$(output Registry) BUCKET=$(output Bucket) INSTANCE=$(output InstanceId)

# The tag is the commit, so the image must be exactly that commit.
[[ -z "$(git status --porcelain -- Dockerfile .dockerignore pyproject.toml uv.lock manifest.csv README.md src deploy)" ]] ||
    { echo "Uncommitted changes: commit them first, the image is tagged with the commit." >&2; exit 1; }
APP_TAG=$(git rev-parse --short=12 HEAD)
# OpenSearch changes only when its Dockerfile does: tag it by that file's content.
OPENSEARCH_TAG=$(git hash-object deploy/opensearch/Dockerfile | cut -c1-12)

aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"

build_and_push() { # <repository> <tag> <build context>; skipped if the registry has it already
    local image="$REGISTRY/$STACK/$1:$2"
    if aws ecr describe-images --repository-name "$STACK/$1" --image-ids "imageTag=$2" >/dev/null 2>&1; then
        echo "== $1:$2 is in the registry already"
        return
    fi
    echo "== building $1:$2"
    # the server is ARM (Graviton); no provenance attestation: it would be a second image in the registry
    docker build --platform linux/arm64 --provenance=false -t "$image" "$3"
    docker push --quiet "$image"
}
build_and_push app "$APP_TAG" .
build_and_push opensearch "$OPENSEARCH_TAG" deploy/opensearch

echo "== uploading deploy files"
for f in docker-compose.prod.yml Caddyfile remote.sh; do
    aws s3 cp --only-show-errors "deploy/$f" "s3://$BUCKET/deploy/$f"
done

echo "== updating the server"
dir=/opt/$STACK
remote="aws s3 cp --only-show-errors s3://$BUCKET/deploy/remote.sh $dir/remote.sh && bash $dir/remote.sh $APP_TAG $OPENSEARCH_TAG"
command_id=$(aws ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
    --comment "deploy $APP_TAG" --timeout-seconds 120 --output text --query Command.CommandId \
    --parameters "{\"executionTimeout\": [\"900\"], \"commands\": [\"$remote\"]}")
result() { aws ssm get-command-invocation --command-id "$command_id" --instance-id "$INSTANCE" \
    --output text --query "$1" 2>/dev/null || true; }
for _ in $(seq 200); do # up to 17 minutes; the first deploy pulls 3 GB of images
    sleep 5
    status=$(result Status)
    case "$status" in "" | Pending | InProgress | Delayed) ;; *) break ;; esac
done
result StandardOutputContent
if [[ "$status" != Success ]]; then
    result StandardErrorContent >&2
    echo "Deploy failed on the server ($status)." >&2
    exit 1
fi
echo "$(output Url) runs $APP_TAG"
