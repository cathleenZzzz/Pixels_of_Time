# Pixel Mosaic + QR (No-Persistence)

Web app that:
- accepts photo uploads (HEIC/JPG/PNG…)
- generates a global pixel mosaic sorted by brightness+hue (golden-ish layout with minimal padding)
- composes mosaic + QR (linking back to the same page) into one PNG
- outputs at 1/4 US Letter size @300DPI: 1275×1650 px
- does not permanently store uploads (RAM + temp directory deleted after request)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
