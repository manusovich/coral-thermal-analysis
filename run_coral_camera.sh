#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

git -C "$ROOT_DIR" pull --ff-only
"$ROOT_DIR/coral_camera" --camera /dev/video1 --output-fps 2 --port 8080
