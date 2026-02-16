#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/apps/donch"
SRC_SERVICE="$ROOT/deploy/systemd/donch-autopar-daily.service"
SRC_TIMER="$ROOT/deploy/systemd/donch-autopar-daily.timer"
DST_DIR="/etc/systemd/system"

if [[ ! -f "$SRC_SERVICE" || ! -f "$SRC_TIMER" ]]; then
  echo "[ERROR] Missing unit files in $ROOT/deploy/systemd" >&2
  exit 1
fi

install -m 0644 "$SRC_SERVICE" "$DST_DIR/donch-autopar-daily.service"
install -m 0644 "$SRC_TIMER" "$DST_DIR/donch-autopar-daily.timer"

systemctl daemon-reload
systemctl enable --now donch-autopar-daily.timer
systemctl start donch-autopar-daily.service
systemctl status --no-pager donch-autopar-daily.timer
