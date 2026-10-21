#!/bin/bash
set -e

SERVICE_NAME="handbook-ai.service"
SERVICE_SRC="/home/mhis/Documents/handbook-ai/${SERVICE_NAME}"
SERVICE_DEST="/etc/systemd/system/${SERVICE_NAME}"

if [ "$EUID" -ne 0 ]; then
  echo "Harap jalankan script ini menggunakan sudo:"
  echo "sudo bash install_service.sh"
  exit 1
fi

echo ">> Menyalin file service ke /etc/systemd/system/..."
cp "$SERVICE_SRC" "$SERVICE_DEST"

echo ">> Menyetel hak akses..."
chmod 644 "$SERVICE_DEST"

echo ">> Reload systemd daemon..."
systemctl daemon-reload

echo ">> Mengaktifkan service agar otomatis jalan saat restart..."
systemctl enable "$SERVICE_NAME"

echo ">> Menjalankan service sekarang..."
systemctl restart "$SERVICE_NAME"

echo ">> Selesai! Cek status service:"
systemctl status "$SERVICE_NAME" --no-pager
