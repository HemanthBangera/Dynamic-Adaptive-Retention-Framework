#!/usr/bin/env bash
# Install the Hugging Face cache once it has finished uploading.
set -euo pipefail
BUCKET="${1:-dars-revision-445527450587}"
export PATH="$HOME/.local/bin:$PATH"
aws s3 cp "s3://$BUCKET/bootstrap/hf.tgz" /tmp/hf.tgz --only-show-errors
mkdir -p "$HOME/.cache"
tar -xzf /tmp/hf.tgz -C "$HOME/.cache"
rm -f /tmp/hf.tgz
du -sh "$HOME/.cache/huggingface"
