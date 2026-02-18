const filesEl = document.getElementById("files");
const goEl = document.getElementById("go");
const statusEl = document.getElementById("status");
const downloadEl = document.getElementById("download");
const hueStartEl = document.getElementById("hueStart");
const skipAlphaEl = document.getElementById("skipAlpha");

function setStatus(msg) {
  statusEl.textContent = msg;
}

goEl.addEventListener("click", async () => {
  downloadEl.style.display = "none";
  downloadEl.href = "";
  const files = filesEl.files;
  if (!files || files.length === 0) {
    setStatus("Please choose some photos first.");
    return;
  }

  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  fd.append("hue_start", hueStartEl.value || "30");
  fd.append("skip_alpha", skipAlphaEl.checked ? "true" : "false");

  setStatus("Uploading & generating… (keep this tab open)");
  goEl.disabled = true;

  try {
    const resp = await fetch("/generate", { method: "POST", body: fd });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(txt || `Server error (${resp.status})`);
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);

    downloadEl.href = url;
    downloadEl.style.display = "inline-block";
    setStatus("Done. Click Download.");
  } catch (e) {
    setStatus("Error: " + e.message);
  } finally {
    goEl.disabled = false;
  }
});
