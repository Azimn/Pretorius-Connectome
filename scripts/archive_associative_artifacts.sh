#!/usr/bin/env bash
# Preserve historical associative-memory result artifacts as permanent Git commits.
# Does not rerun the biological experiments or modify their original records.
set -euo pipefail

: "${GH_TOKEN:?A repository-scoped GitHub token is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
workspace="${RUNNER_TEMP:-/tmp}/associative-archive"
mkdir -p "$workspace"

archive_file() {
  local run="$1" artifact="$2" filename="$3" destination="$4"
  local directory="$workspace/$run/$artifact"
  mkdir -p "$directory"
  echo "::group::Archive $artifact ($run) -> $destination"
  # Multiple source JSON files can share one artifact ZIP; reuse one extraction.
  if ! find "$directory" -type f -print -quit | grep -q .; then
    gh run download "$run" --repo "$GITHUB_REPOSITORY" --name "$artifact" --dir "$directory"
  fi
  mapfile -t files < <(find "$directory" -type f -name "$filename" -print)
  if [[ "${#files[@]}" -ne 1 ]]; then
    printf 'Expected one %s file, got %s\n' "$filename" "${#files[@]}" >&2
    exit 1
  fi
  local source="${files[0]}"
  python - "$source" <<'PY'
import json, sys
from pathlib import Path
file = Path(sys.argv[1])
data = json.loads(file.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit("Expected a JSON object")
print(f"Verified JSON: {file.name}, {len(file.read_bytes())} bytes, top-level keys: {len(data)}")
PY

  if gh api "repos/$GITHUB_REPOSITORY/contents/$destination" --silent >/dev/null 2>&1; then
    echo "Existing permanent archive detected; preserving without overwrite."
  else
    local request="$workspace/payload_${run}_$(basename "$destination").json"
    python - "$source" "$request" "$run" "$artifact" <<'PY'
import base64, json, sys
from pathlib import Path
source, payload, run, artifact = sys.argv[1:]
body = {
  "message": f"Archive complete experimental JSON from GitHub Actions run {run}: {artifact}",
  "branch": "main",
  "content": base64.b64encode(Path(source).read_bytes()).decode("ascii"),
}
Path(payload).write_text(json.dumps(body), encoding="utf-8")
PY
    gh api --method PUT "repos/$GITHUB_REPOSITORY/contents/$destination" --input "$request" \
      --jq '.commit.sha'
  fi
  echo "::endgroup::"
}

archive_file 37833271857 associative-synthetic-diagnostic synthetic-posthoc.json \
  results/associative/runs/synthetic-seed31-run37833271857.json
archive_file 37833431615 associative-flywire-v783-diagnostic flywire-v783-posthoc.json \
  results/associative/runs/flywire-full-seed31-run37833431615.json
archive_file 37836107318 flywire-associative-three-seed-posthoc flywire-v783-three-seed.json \
  results/associative/runs/flywire-full-three-seed-run37836107318.json
archive_file 37836674947 flywire-mb-region-three-seed-posthoc flywire-v783-mb-three-seed.json \
  results/associative/runs/flywire-mb-three-seed-run37836674947.json
archive_file 37836674947 flywire-mb-region-three-seed-posthoc results_mb_conversion.json \
  results/associative/runs/flywire-mb-conversion-run37836674947.json

echo "Completed archival of all five named historical associative result files."
