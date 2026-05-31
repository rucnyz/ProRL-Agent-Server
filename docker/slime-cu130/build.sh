#!/usr/bin/env bash
# Build slimerl/slime-cu130:local for B300 (sm_103a).
# This requires slime/ to be a sibling git checkout — its docker context is sent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"

# Make sure our patched Dockerfile is in the Slime checkout.
cp "${HERE}/Dockerfile" "${SLIME_DIR}/docker/Dockerfile"

cd "${SLIME_DIR}"
docker build \
  --build-arg ENABLE_CUDA_13=1 \
  --build-arg ENABLE_SGLANG_PATCH=0 \
  --build-arg SGLANG_IMAGE_TAG=v0.5.12.post1-cu129 \
  --build-arg MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862 \
  --build-arg PATCH_VERSION=latest \
  -t slimerl/slime-cu130:local \
  -f docker/Dockerfile \
  .
