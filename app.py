from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
from your_asl_code import run_enhanced_asl_detection, FingerSpellingEngine, load_resnet_model, preprocess_frame, ASL_CLASSES
import torch

app = Flask(__name__)

# Initialize model & spelling engine once
model = load_resnet_model("models/asl_resnet_model.pth")
spelling_engine = FingerSpellingEngine()
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        # ROI
        h, w = frame.shape[:2]
        roi = frame[h//2-150:h//2+150, w//2-150:w//2+150]

        # Predict
        try:
            input_tensor = preprocess_frame(roi)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = ASL_CLASSES[predicted.item()]
                confidence_score = confidence.item()

            spelling_engine.process_prediction(predicted_class, confidence_score)
        except:
            predicted_class = "error"
            confidence_score = 0

        # Draw ROI
        cv2.rectangle(frame, (w//2-150, h//2-150), (w//2+150, h//2+150), (0, 255, 0), 2)

        # Convert frame for web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(spelling_engine.get_status())

@app.route('/action', methods=['POST'])
def action():
    data = request.json
    if data['cmd'] == 'clear':
        spelling_engine.clear_all()
    elif data['cmd'] == 'undo':
        spelling_engine.undo_last_word()
    elif data['cmd'] == 'save':
        spelling_engine.save_session()
    return jsonify({"msg": "Action done"})

if __name__ == "__main__":
    app.run(debug=True)
