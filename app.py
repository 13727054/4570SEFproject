from flask import Flask, request, jsonify
from PIL import Image
from model import load_pipeline
import os

app = Flask(__name__)
pipe = load_pipeline()

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form.get("prompt", "")
    negative_prompt = request.form.get("negative_prompt", "")
    strength = float(request.form.get("strength", 0.9))

    # 读取上传的图片
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files["image"]).convert("RGB").resize((1024, 1024))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=7.5
    )
    result.images[0].save("output.png")

    return jsonify({"message": "done", "file": "output.png"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
