#!/bin/bash
# Runs ON THE SERVER, as root, started by deploy/deploy.sh through Systems Manager:
#   remote.sh <app image tag> <opensearch image tag>
# Brings the server to that version: deploy files, documents, secrets, images, search index.
# Safe to run again; it only redoes what changed.
set -euo pipefail

APP_TAG=$1 OPENSEARCH_TAG=$2
cd "$(dirname "$0")"
source ./stack.env # AWS_REGION, BUCKET, REGISTRY, DOMAIN, SECRETS_PATH: written at first boot
export AWS_DEFAULT_REGION=$AWS_REGION
export HOME=/root # Systems Manager starts commands without one; docker login needs it

echo "== deploy files"
aws s3 cp --only-show-errors "s3://$BUCKET/deploy/docker-compose.prod.yml" docker-compose.yml
aws s3 cp --only-show-errors "s3://$BUCKET/deploy/Caddyfile" Caddyfile
printf 'REGISTRY=%s\nIMAGE_TAG=%s\nOPENSEARCH_TAG=%s\nDOMAIN=%s\n' "$REGISTRY" "$APP_TAG" "$OPENSEARCH_TAG" "$DOMAIN" > .env

echo "== documents"
# One archive, not a folder sync: it keeps the files' times. The app re-parses a PDF that
# looks newer than its cached pages, and it can't write the result to its read-only data/.
new_data=$(aws s3api head-object --bucket "$BUCKET" --key data.tar.gz --query ETag --output text)
if [[ "$new_data" != "$(cat .data-version 2>/dev/null || true)" ]]; then
    aws s3 cp --only-show-errors "s3://$BUCKET/data.tar.gz" data.tar.gz
    rm -rf data.new && mkdir data.new && tar -xzf data.tar.gz -C data.new
    chown -R 1000:1000 data.new # the containers' user
    rm -rf data && mv data.new data && rm data.tar.gz
    rm -f .indexed-version
    echo "new documents: $(ls data/*.pdf | wc -l) PDFs"
else
    echo "unchanged"
fi

echo "== secrets"
# Every parameter under $SECRETS_PATH becomes an environment variable of the app.
# Values are single-quoted, so Compose takes them literally (a "$" stays a "$").
aws ssm get-parameters-by-path --path "$SECRETS_PATH" --with-decryption --output json | python3 -c '
import json, sys
lines = []
for p in json.load(sys.stdin)["Parameters"]:
    name, value = p["Name"].rsplit("/", 1)[1], p["Value"]
    if "\x27" in value or "\n" in value:
        sys.exit(f"{name}: a value with a single quote or a line break can not be passed on")
    lines.append(f"{name}=\x27{value}\x27\n")
open("secrets.env", "w").writelines(lines)
'
chmod 600 secrets.env
grep -q '^APP_PASSWORD=' secrets.env || {
    echo "No $SECRETS_PATH/APP_PASSWORD in Parameter Store: refusing to start a public site without a login." >&2
    exit 1
}
grep -q '^ANTHROPIC_API_KEY=' secrets.env || echo "WARNING: no $SECRETS_PATH/ANTHROPIC_API_KEY yet: the site starts, but can't answer."

echo "== images"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" 2>/dev/null
docker compose pull --quiet
docker compose up -d --remove-orphans --wait opensearch

echo "== search index"
# rebuilt when the documents changed, the app image changed (chunk fields may have), or it's missing
index_version="$new_data $APP_TAG"
if [[ "$index_version" != "$(cat .indexed-version 2>/dev/null || true)" ]] ||
    ! docker compose exec -T opensearch curl -sf -o /dev/null localhost:9200/etf-chunks; then
    docker compose run --rm indexer 2>&1 | grep -v onnxruntime
    echo "$new_data" > .data-version
    echo "$index_version" > .indexed-version
else
    echo "unchanged"
fi

echo "== start"
docker compose up -d --remove-orphans
for _ in $(seq 60); do # the app answers /health with 200 once it can search
    if docker compose exec -T app python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" 2>/dev/null; then
        docker compose ps --format 'table {{.Service}}\t{{.Image}}\t{{.Status}}'
        docker image prune -af >/dev/null # images of earlier versions
        echo "deployed $APP_TAG"
        exit 0
    fi
    sleep 3
done
echo "The app did not become healthy:" >&2
docker compose logs --tail 40 app >&2
exit 1
