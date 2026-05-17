from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import io
import time
from collections import deque, Counter
import os
import json

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# ASL Classes
# -------------------------------
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# -------------------------------
# Model Utilities
# -------------------------------
def create_resnet_model(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(model_path="models/asl_resnet_model.pth"):
    try:
        model = create_resnet_model(len(ASL_CLASSES))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# -------------------------------
# Image Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def preprocess_frame(frame, image_size=224):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

# -------------------------------
# Finger Spelling Engine
# -------------------------------
class FingerSpellingEngine:
    def __init__(self):
        self.current_word = ""
        self.sentence = ""
        self.word_history = []
        self.last_prediction = ""
        self.last_prediction_time = 0
        self.prediction_hold_time = 1.5
        self.delete_hold_time = 2.0
        self.space_hold_time = 1.0
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.75

    def smooth_prediction(self, predicted_class, confidence):
        self.prediction_buffer.append((predicted_class, confidence))
        high_conf_predictions = [pred for pred, conf in self.prediction_buffer if conf > self.confidence_threshold]
        if len(high_conf_predictions) < 3:
            return None, 0
        counter = Counter(high_conf_predictions)
        most_common_pred, _ = counter.most_common(1)[0]
        avg_conf = np.mean([conf for pred, conf in self.prediction_buffer if pred == most_common_pred])
        return most_common_pred, avg_conf

    def process_prediction(self, predicted_class, confidence):
        current_time = time.time()
        smoothed_pred, smoothed_conf = self.smooth_prediction(predicted_class, confidence)
        if smoothed_pred is None:
            return
        if smoothed_pred == self.last_prediction:
            hold_time = current_time - self.last_prediction_time
            if smoothed_pred == 'del' and hold_time > self.delete_hold_time:
                self.handle_delete(); self.last_prediction_time = current_time
            elif smoothed_pred == 'space' and hold_time > self.space_hold_time:
                self.handle_space(); self.last_prediction_time = current_time
            elif smoothed_pred not in ['del', 'space', 'nothing'] and hold_time > self.prediction_hold_time:
                self.current_word += smoothed_pred.lower(); self.last_prediction_time = current_time
        else:
            self.last_prediction = smoothed_pred
            self.last_prediction_time = current_time

    def handle_delete(self):
        if self.current_word: self.current_word = self.current_word[:-1]
        elif self.sentence: self.sentence = self.sentence[:-1]

    def handle_space(self):
        if self.current_word:
            if self.sentence: self.sentence += " " + self.current_word
            else: self.sentence = self.current_word
            self.word_history.append(self.current_word)
            self.current_word = ""

    def get_status(self):
        return {'current_word': self.current_word, 'sentence': self.sentence}

# -------------------------------
# Flask Prediction API
# -------------------------------
prediction_history = deque(maxlen=15)

def get_prediction(image_bytes):
    tensor = preprocess_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    prediction_history.append(predicted.item())
    most_common = Counter(prediction_history).most_common(1)[0][0]
    return ASL_CLASSES[most_common], confidence.item()

@app.route("/")
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head><title>ASL Detection</title></head>
        <body>
            <h1>ASL Detection</h1>
            <video id="video" width="400" height="300" autoplay></video>
            <div id="prediction">Prediction: None</div>
            <button onclick="startDetection()">Start Detection</button>
            <script>
                const video=document.getElementById('video');
                const predictionDiv=document.getElementById('prediction');
                let detecting=false;
                async function startDetection(){
                    detecting=true;
                    const stream=await navigator.mediaDevices.getUserMedia({video:true});
                    video.srcObject=stream;
                    const canvas=document.createElement('canvas');
                    const ctx=canvas.getContext('2d');
                    async function captureFrame(){
                        if(!detecting) return;
                        canvas.width=video.videoWidth;
                        canvas.height=video.videoHeight;
                        ctx.drawImage(video,0,0,canvas.width,canvas.height);
                        canvas.toBlob(async(blob)=>{
                            const formData=new FormData();
                            formData.append('file',blob,'frame.jpg');
                            try{
                                const res=await fetch('/predict',{method:'POST',body:formData});
                                const result=await res.json();
                                predictionDiv.innerText="Prediction: "+result.prediction+
                                    " ("+(result.confidence*100).toFixed(1)+"%)";
                            }catch(err){console.error(err);}
                        },'image/jpeg');
                        setTimeout(captureFrame,500);
                    }
                    captureFrame();
                }
            </script>
        </body>
        </html>
    """)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    img_bytes = request.files["file"].read()
    prediction, confidence = get_prediction(img_bytes)
    return jsonify({"prediction": prediction, "confidence": confidence})

# -------------------------------
# OpenCV Finger Spelling Runner
# -------------------------------
def run_enhanced_asl_detection():
    cap = cv2.VideoCapture(0)
    spelling_engine = FingerSpellingEngine()
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)
        roi=frame[100:400,100:400]
        try:
            tensor=preprocess_frame(roi)
            with torch.no_grad():
                outputs=model(tensor)
                probs=torch.softmax(outputs,dim=1)
                conf,pred=torch.max(probs,1)
                predicted_class=ASL_CLASSES[pred.item()]
                confidence_score=conf.item()
            spelling_engine.process_prediction(predicted_class,confidence_score)
        except: predicted_class="error"; confidence_score=0
        cv2.rectangle(frame,(100,100),(400,400),(0,255,0),2)
        cv2.putText(frame,f"{predicted_class} ({confidence_score:.2f})",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame,f"Sentence: {spelling_engine.sentence}",(50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.imshow("Enhanced ASL Detection",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__=="__main__":
    if model:
        print("Model loaded ✅")
        mode=input("Enter mode: 1=Flask Web, 2=OpenCV Camera : ")
        if mode=="1":
            app.run(host="0.0.0.0", port=5000, debug=False)
        else:
            run_enhanced_asl_detection()
    else:
        print("❌ Model not loaded. Check .pth file.")
