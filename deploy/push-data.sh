#!/bin/bash
# Upload the documents to the server's bucket: AWS_PROFILE=... deploy/push-data.sh
# Run it after `uv run ingest` and `uv run index` whenever the PDFs change, then deploy:
# the next deploy unpacks them on the server and rebuilds the search index.
# Not in the archive: data/facts, the evals' answer key. The server has no use for it.
set -euo pipefail
cd "$(dirname "$0")/.."

for needed in data/chunks.jsonl data/catalog.json data/parsed data/embedding_cache; do
    [[ -e $needed ]] || { echo "$needed is missing: run 'uv run ingest' and 'uv run index' first." >&2; exit 1; }
done
missing=$(tail -n +2 manifest.csv | cut -d, -f6 | tr -d '\r' | while read -r f; do [[ -f "data/$f" ]] || echo "$f"; done)
[[ -z "$missing" ]] || { echo "PDFs listed in manifest.csv but not in data/:" $missing >&2; exit 1; }

BUCKET=$(aws cloudformation describe-stacks --stack-name etf-assistant --output text \
    --query "Stacks[0].Outputs[?OutputKey=='Bucket'].OutputValue")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
archive=$tmp/data.tar.gz
# an archive keeps the files' times (see remote.sh); no macOS extras, Linux tar doesn't know them
COPYFILE_DISABLE=1 tar -czf "$archive" --no-xattrs --no-mac-metadata \
    --exclude facts --exclude .DS_Store -C data .
aws s3 cp --only-show-errors "$archive" "s3://$BUCKET/data.tar.gz"
echo "uploaded $(du -h "$archive" | cut -f1) to s3://$BUCKET/data.tar.gz. It goes live with the next deploy."
