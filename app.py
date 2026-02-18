import os
from io import BytesIO

from flask import Flask, render_template, request, send_file, jsonify
import qrcode
from qrcode.image.pil import PilImage

from mosaic import (
    generate_sorted_mosaic,
    compose_quarter_letter_with_qr,
    LETTER_QUARTER_PX,
)
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Hard limits (tune for your deployment)
MAX_FILES = 60
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB request cap

# Pixel caps
# Full mode: allow up to ~120M pixels (still heavy but feasible on decent machines)
FULL_MODE_MAX_PIXELS = 120_000_000
# Fast mode: cap to e.g. 15M sampled pixels (stable + fast on typical servers)
FAST_MODE_SAMPLE_PIXELS = 15_000_000


@app.get("/")
def index():
    return render_template(
        "index.html",
        max_files=MAX_FILES,
        full_cap=FULL_MODE_MAX_PIXELS,
        fast_cap=FAST_MODE_SAMPLE_PIXELS,
        out_w=LETTER_QUARTER_PX[0],
        out_h=LETTER_QUARTER_PX[1],
    )


@app.post("/generate")
def generate():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    if len(files) > MAX_FILES:
        return jsonify({"error": f"Too many photos. Limit is {MAX_FILES}."}), 400

    # Read all bytes into memory; do NOT persist originals
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
    fast_mode = request.form.get("fast_mode", "false").lower() == "true"

    # Safe cap logic:
    # - We estimate/count pixels inside generate_sorted_mosaic.
    # - If too large and not fast_mode => return a warning/error with instructions.
    try:
        mosaic_img, meta = generate_sorted_mosaic(
            file_bytes_list=file_bytes_list,
            skip_alpha=skip_alpha,
            hue_start_degrees=hue_start,
            full_mode_cap_pixels=FULL_MODE_MAX_PIXELS,
            fast_mode=fast_mode,
            fast_mode_sample_pixels=FAST_MODE_SAMPLE_PIXELS,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

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
    qr_img = qr.make_image(
        fill_color="black", back_color="white", image_factory=PilImage
    ).convert("RGB")

    combined = compose_quarter_letter_with_qr(mosaic_img, qr_img, target_px=LETTER_QUARTER_PX)

    # Return as PNG
    bio = BytesIO()
    combined.save(bio, format="PNG", optimize=True)
    bio.seek(0)

    resp = send_file(
        bio,
        mimetype="image/png",
        as_attachment=True,
        download_name="mosaic_with_qr.png",
    )

    # Optional: pass info back via headers (nice for debugging)
    resp.headers["X-Mosaic-Mode"] = "fast" if meta["mode"] == "fast" else "full"
    resp.headers["X-Mosaic-Input-Pixels"] = str(meta["input_pixels"])
    resp.headers["X-Mosaic-Used-Pixels"] = str(meta["used_pixels"])
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
