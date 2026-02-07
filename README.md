# Coral Thermal Analysis

## MJPEG Stream (coral_camera)

Run locally:
```
./run_coral_camera.sh
```

Stream URL:
```
http://<coral-ip>:8080/stream
```

### Enable at boot (systemd)

Install and start the service:
```
./install_coral_camera_service.sh
```

Service management:
```
systemctl status coral-camera.service
journalctl -u coral-camera.service -f
sudo systemctl restart coral-camera.service
sudo systemctl stop coral-camera.service
```

If your repo path differs from `/home/mendel/coral-thermal-analysis`, edit `install_coral_camera_service.sh` before running it.
