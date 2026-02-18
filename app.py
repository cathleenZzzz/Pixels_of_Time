import os
from io import BytesIO

from flask import Flask, render_template, request, send_file, jsonify
import qrcode
from qrcode.image.pil import PilImage

from mosaic import generate_sorted_mosaic, compose_quarter_letter_with_qr, LETTER_QUARTER_PX

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB upload cap (adjust)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/generate")
def generate():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    # Read all bytes into memory; do NOT save to disk permanently
    file_bytes_list = []
    for f in files:
        b = f.read()
        if b:
            file_bytes_list.append(b)

    if not file_bytes_list:
        return jsonify({"error": "Uploaded files were empty"}), 400

    # Options
    hue_start = float(request.form.get("hue_start", 30.0))
    skip_alpha = request.form.get("skip_alpha", "true").lower() == "true"

    # Build mosaic (golden-ish, minimal padding, global sort)
    mosaic_img = generate_sorted_mosaic(
        file_bytes_list=file_bytes_list,
        skip_alpha=skip_alpha,
        hue_start_degrees=hue_start,
    )

    # QR points back to this page (same host)
    page_url = request.url_root.rstrip("/") + "/"
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(page_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white", image_factory=PilImage).convert("RGB")

    # Compose into 1/4-letter-sized image
    combined = compose_quarter_letter_with_qr(mosaic_img, qr_img, target_px=LETTER_QUARTER_PX)

    # Return as PNG in-memory
    bio = BytesIO()
    combined.save(bio, format="PNG", optimize=True)
    bio.seek(0)

    return send_file(
        bio,
        mimetype="image/png",
        as_attachment=True,
        download_name="mosaic_with_qr.png",
    )


if __name__ == "__main__":
    # For local dev:
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
