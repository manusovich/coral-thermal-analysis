#!/usr/bin/env bash
set -euo pipefail

SERVICE_PATH="/etc/systemd/system/coral-camera.service"
APP_DIR="/home/mendel/coral-thermal-analysis"
RUN_SCRIPT="$APP_DIR/run_coral_camera.sh"

sudo tee "$SERVICE_PATH" > /dev/null <<SERVICE
[Unit]
Description=Coral Camera MJPEG Stream
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$RUN_SCRIPT
Restart=always
RestartSec=2
User=mendel

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable coral-camera.service
sudo systemctl start coral-camera.service

echo "Installed and started coral-camera.service"
