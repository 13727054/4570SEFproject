from flask import Flask, request, jsonify
from PIL import Image
from model import load_pipeline

app = Flask(__name__)
pipe = load_pipeline()

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    image = Image.open(request.files["image"]).convert("RGB").resize((1024, 1024))
    result = pipe(prompt=prompt, image=image, strength=0.9, guidance_scale=7.5)
    result.images[0].save("output.png")
    return jsonify({"message": "done", "file": "output.png"})
