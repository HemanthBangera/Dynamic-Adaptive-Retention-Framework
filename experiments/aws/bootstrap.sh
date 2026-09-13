#!/usr/bin/env bash
# Prepare the EC2 host for the DARS test phase: fetch the payload, build the pinned
# environment, and prove it matches the laptop before anything is run for real.
set -euo pipefail
BUCKET="${1:-dars-revision-445527450587}"
HOME_DIR="$HOME"
REPO="$HOME_DIR/dars/Dynamic-Adaptive-Retention-Framework"

echo "== packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential curl unzip jq >/dev/null

echo "== aws cli"
if ! command -v aws >/dev/null 2>&1; then
  curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install >/dev/null
fi

echo "== payload from s3"
mkdir -p "$HOME_DIR/dars" "$HOME_DIR/.cache"
for f in repo.tgz caches.tgz; do
  aws s3 cp "s3://$BUCKET/bootstrap/$f" "/tmp/$f" --only-show-errors
done
tar -xzf /tmp/repo.tgz   -C "$HOME_DIR/dars"
tar -xzf /tmp/caches.tgz -C "$REPO"
# The Hugging Face cache is large and may still be uploading; fetch it when it lands.
if aws s3api head-object --bucket "$BUCKET" --key bootstrap/hf.tgz >/dev/null 2>&1; then
  aws s3 cp "s3://$BUCKET/bootstrap/hf.tgz" /tmp/hf.tgz --only-show-errors
  tar -xzf /tmp/hf.tgz -C "$HOME_DIR/.cache"
  echo "hugging face cache installed"
else
  echo "WARNING: hf.tgz not in the bucket yet; run experiments/aws/fetch_hf.sh once it is"
fi
rm -f /tmp/*.tgz

echo "== python 3.14.3 via uv"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi
export PATH="$HOME_DIR/.local/bin:$PATH"
cd "$REPO"
uv python install 3.14.3
uv venv --python 3.14.3 .venv >/dev/null

# experiments/requirements-vm.txt pins every package this project imports to the version in
# the frozen laptop lock file. We do not install the laptop's whole global freeze: it contains
# unrelated packages whose pins contradict each other (awscli vs botocore, streamlit vs pyarrow)
# and others with no cp314 wheel that would try to build against X11 headers. torch comes from
# the PyTorch CPU index so we get the same CPU build as the laptop.
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
uv pip install --python .venv/bin/python -r experiments/requirements-vm.txt
uv pip freeze --python .venv/bin/python > experiments/requirements-vm-installed.txt

echo "== environment check"
.venv/bin/python - <<'PY'
import torch, sentence_transformers, tiktoken, qdrant_client, openai, numpy, platform, sys
print("python    ", sys.version.split()[0])
print("platform  ", platform.platform())
print("torch     ", torch.__version__)
print("sentence_t", sentence_transformers.__version__)
print("qdrant    ", qdrant_client.__version__ if hasattr(qdrant_client, "__version__") else "?")
print("openai    ", openai.__version__)
print("numpy     ", numpy.__version__)
tiktoken.encoding_for_model("gpt-4o-mini")   # pull the encoding once, while the network is known good
print("tiktoken encoding ok")
PY
echo "BOOTSTRAP OK"
