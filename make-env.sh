#!/usr/bin/env bash
# Generate this repo's git-ignored .env by copying a source env file kept outside
# the repo:  ./make-env.sh [source.env]
#
# The source defaults to ../../fine-tuning.env, for a checkout nested two levels
# inside a project folder that holds the real values. .env is derived; edit the
# source, not .env. See .env.example for the variables.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="${1:-$here/../../fine-tuning.env}"
dest="$here/.env"

if [[ ! -f "$src" ]]; then
  echo "source env file not found: $src" >&2
  echo "pass a path (./make-env.sh /path/to/source.env)," >&2
  echo "or copy .env.example to .env and fill it in by hand." >&2
  exit 1
fi

cp "$src" "$dest"
echo "wrote $dest"
echo "  from $src"
