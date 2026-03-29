from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from fusion import fusion_decision
from audio_model import predict_audio

app = Flask(__name__)
CORS(app)

# -----------------------
# IMAGE MODEL (Mock)
# -----------------------
@app.route('/detect/image', methods=['POST'])
def detect_image():
    file = request.files['file']
    score = random.uniform(0, 1)
    return jsonify({
        "model": "Image",
        "score": round(score, 3),
        "result": "FAKE" if score > 0.5 else "REAL"
    })


# -----------------------
# VIDEO MODEL (Mock)
# -----------------------
@app.route('/detect/video', methods=['POST'])
def detect_video():
    file = request.files['file']
    score = random.uniform(0, 1)
    return jsonify({
        "model": "Video",
        "score": round(score, 3),
        "result": "FAKE" if score > 0.5 else "REAL"
    })


# -----------------------
# AUDIO MODEL (Mock)
# -----------------------
@app.route("/detect/audio", methods=["POST"])
def detect_audio():
    file = request.files["file"]

    path = "temp_audio.wav"
    file.save(path)

    result = predict_audio(path)

    return jsonify(result)


# -----------------------
# FUSION MODEL
# -----------------------
@app.route('/detect/fusion', methods=['POST'])
def detect_fusion():
    data = request.json
    image_score = data.get("image_score", 0)
    video_score = data.get("video_score", 0)
    audio_score = data.get("audio_score", 0)

    result = fusion_decision(image_score, video_score, audio_score)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)