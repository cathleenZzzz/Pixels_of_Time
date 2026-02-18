const filesEl = document.getElementById("files");
const goEl = document.getElementById("go");
const statusEl = document.getElementById("status");
const downloadEl = document.getElementById("download");
const hueStartEl = document.getElementById("hueStart");
const skipAlphaEl = document.getElementById("skipAlpha");
const fastModeEl = document.getElementById("fastMode");

const MAX_FILES = window.__MAX_FILES__ || 60;

function setStatus(msg) {
  statusEl.textContent = msg;
}

filesEl.addEventListener("change", () => {
  const n = filesEl.files?.length || 0;
  if (n > MAX_FILES) {
    setStatus(`You selected ${n} photos. The limit is ${MAX_FILES}. Please remove some.`);
  } else {
    setStatus("");
  }
});

goEl.addEventListener("click", async () => {
  downloadEl.style.display = "none";
  downloadEl.href = "";

  const files = filesEl.files;
  if (!files || files.length === 0) {
    setStatus("Please choose some photos first.");
    return;
  }
  if (files.length > MAX_FILES) {
    setStatus(`Too many photos (${files.length}). Limit is ${MAX_FILES}.`);
    return;
  }

  const fd = new FormData();
  for (const f of files) fd.append("files", f);

  fd.append("hue_start", hueStartEl.value || "30");
  fd.append("skip_alpha", skipAlphaEl.checked ? "true" : "false");
  fd.append("fast_mode", fastModeEl.checked ? "true" : "false");

  setStatus("Uploading & generating… (keep this tab open)");
  goEl.disabled = true;

  try {
    const resp = await fetch("/generate", { method: "POST", body: fd });

    // If server returns JSON error
    const contentType = resp.headers.get("content-type") || "";
    if (!resp.ok) {
      if (contentType.includes("application/json")) {
        const j = await resp.json();
        throw new Error(j.error || `Server error (${resp.status})`);
      } else {
        const txt = await resp.text();
        throw new Error(txt || `Server error (${resp.status})`);
      }
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);

    downloadEl.href = url;
    downloadEl.style.display = "inline-block";

    const mode = resp.headers.get("X-Mosaic-Mode");
    const inputPx = resp.headers.get("X-Mosaic-Input-Pixels");
    const usedPx = resp.headers.get("X-Mosaic-Used-Pixels");
    if (mode && inputPx && usedPx) {
      setStatus(`Done (${mode} mode). Input pixels: ${inputPx}. Used pixels: ${usedPx}. Click Download.`);
    } else {
      setStatus("Done. Click Download.");
    }
  } catch (e) {
    setStatus("Error: " + e.message);
  } finally {
    goEl.disabled = false;
  }
});
