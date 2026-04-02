from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from main import extract_room_labels, fill_wall_gaps, detect_rooms, match_room_and_label, extract_walls_data

app = Flask(__name__)

# ── 完整 CORS 配置：允许所有来源的跨域请求（含预检 OPTIONS）──
CORS(app, resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/upload_floorplan', methods=['POST', 'OPTIONS'])
def upload_floorplan():
    # 处理预检请求
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        room_labels, text_extracted_img = extract_room_labels(img)
        gaps_filled_img = fill_wall_gaps(text_extracted_img)
        _, initial_contours, segmented_img = detect_rooms(img, gaps_filled_img, text_extracted_img)
        rooms_data, _ = match_room_and_label(room_labels, initial_contours, segmented_img)
        walls_data = extract_walls_data(img)

        return jsonify({
            "status": "success",
            "data": {
                "rooms": rooms_data,
                "walls": walls_data
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)