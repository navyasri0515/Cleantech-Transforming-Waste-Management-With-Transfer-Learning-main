import os, uuid, numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ── Flask setup ─────────────────────────────────────────────────────────
app = Flask(__name__)

# folder for user uploads
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── load trained VGG-16 model once at startup ───────────────────────────
MODEL_PATH = "Vgg16.h5"
CLASSES = ["Biodegradable", "Recyclable", "Trash"]

model = load_model(MODEL_PATH)
print(f"[INFO] Model {MODEL_PATH} loaded. Classes: {CLASSES}")

# ── helper: run prediction on one image ─────────────────────────────────
def predict(path: str):
    img = load_img(path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    probs = model.predict(arr[np.newaxis])[0]               # e.g. [0.12, 0.78, 0.10]
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), dict(zip(CLASSES, probs))

# ── ROUTES ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")              # hero + about on one page

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/portfolio")
def portfolio():
    return render_template("portfolio_details.html") # new case-study page

@app.route("/classify", methods=["GET", "POST"])
def classify():
    """
    GET  → show upload form.
    POST → save uploaded image, run model, show result page.
    """
    if request.method == "POST" and "file" in request.files:
        f = request.files["file"]
        if f.filename:
            filename = f"{uuid.uuid4().hex}_{f.filename}"
            save_path = os.path.join(UPLOAD_DIR, filename)
            f.save(save_path)

            label, conf, probs = predict(save_path)
            return render_template(
                "result.html",
                filename=filename,
                label=label,
                conf=f"{conf*100:.2f} %",
                probs={k: f"{v*100:.1f} %" for k, v in probs.items()},
                classes=CLASSES,
            )
    # GET fallback
    return render_template("classify.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    # expose user uploads for <img src="...">
    return redirect(url_for("static", filename=f"uploads/{filename}"), code=301)

# custom 404 page
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# ── run local dev server ────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=True auto-reloads on code change; switch to False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
