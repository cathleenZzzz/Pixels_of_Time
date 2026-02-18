# A Pixels Arranging Tool

- accepts photos (HEIC/JPG/PNG/etc)
- generates a mosaic of pixels sorted by brightness+hue, using all the pixels in uploaded images
- outputs at 1/4 US Letter size @300DPI: 1275×1650 px
- does not permanently store your uploads (RAM + temp directory deleted after request)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
