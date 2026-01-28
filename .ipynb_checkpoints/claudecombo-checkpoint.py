import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import mediapipe as mp
from collections import deque
import threading
import queue

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# =====================================
# MODEL DEFINITIONS (from your code)
# =====================================

class SimpleCNN(nn.Module):
    """Simple CNN for ASL recognition"""
    def __init__(self, num_classes=29):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),  # Assuming 224x224 input
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FastViT(nn.Module):
    """Fast Vision Transformer for real-time inference"""
    def __init__(self, img_size=64, patch_size=8, num_classes=29,
                 embed_dim=128, depth=3, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return self.head(x[:, 0])

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x

# =====================================
# REAL-TIME DETECTION CLASS
# =====================================

class RealTimeASLDetector:
    def __init__(self, model_type='cnn', model_path=None, confidence_threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Adjust based on your model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        self.word_buffer = []
        self.last_prediction = 'nothing'
        self.stable_frames = 0
        self.min_stable_frames = 15
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.inference_times = deque(maxlen=30)

    def load_model(self, model_path=None):
        """Load the trained model"""
        if self.model_type == 'cnn':
            model = SimpleCNN(num_classes=29)
        elif self.model_type == 'vit':
            model = FastViT(num_classes=29)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if model_path and torch.cuda.is_available():
            try:
                model.load_state_dict(torch.load(model_path))
                print(f"✅ Loaded model from {model_path}")
            except:
                print(f"⚠️ Could not load model from {model_path}, using random weights")
        
        model.to(self.device)
        model.eval()
        return model

    def extract_hand_region(self, frame):
        """Extract hand region using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                
                h, w = frame.shape[:2]
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                
                # Add padding
                padding = 50
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Extract hand region
                hand_region = frame[y_min:y_max, x_min:x_max]
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                return hand_region, (x_min, y_min, x_max, y_max)
        
        return None, None

    def predict_sign(self, hand_region):
        """Predict ASL sign from hand region"""
        if hand_region is None or hand_region.size == 0:
            return 'nothing', 0.0
        
        start_time = time.time()
        
        try:
            # Preprocess image
            if len(hand_region.shape) == 3 and hand_region.shape[2] == 3:
                input_tensor = self.transform(hand_region).unsqueeze(0).to(self.device)
            else:
                return 'nothing', 0.0
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            predicted_class = ASL_CLASSES[predicted.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 'nothing', 0.0

    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions over multiple frames"""
        if confidence < self.confidence_threshold:
            prediction = 'nothing'
        
        self.prediction_history.append(prediction)
        
        # Get most common prediction
        if len(self.prediction_history) >= 5:
            most_common = max(set(self.prediction_history), key=self.prediction_history.count)
            
            if most_common == self.last_prediction:
                self.stable_frames += 1
            else:
                self.stable_frames = 0
                self.last_prediction = most_common
            
            if self.stable_frames >= self.min_stable_frames:
                return most_common
        
        return self.last_prediction

    def update_word_buffer(self, stable_prediction):
        """Update word buffer for sentence construction"""
        if stable_prediction == 'space':
            if self.word_buffer:
                self.word_buffer.append(' ')
        elif stable_prediction == 'del':
            if self.word_buffer:
                self.word_buffer.pop()
        elif stable_prediction != 'nothing' and stable_prediction != self.last_prediction:
            self.word_buffer.append(stable_prediction)
            self.stable_frames = 0  # Reset for next character

    def draw_ui(self, frame, prediction, confidence, stable_prediction):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Current prediction
        color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
        cv2.putText(frame, f"Current: {prediction} ({confidence:.2f})", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Stable prediction
        cv2.putText(frame, f"Stable: {stable_prediction}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Current word
        current_word = ''.join(self.word_buffer[-20:])  # Show last 20 characters
        cv2.putText(frame, f"Word: {current_word}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)
        
        # Performance info
        if self.inference_times:
            avg_inference = np.mean(self.inference_times) * 1000
            fps = 1.0 / max(np.mean(self.inference_times), 0.001)
            cv2.putText(frame, f"FPS: {fps:.1f} | Inference: {avg_inference:.1f}ms", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'c' to clear word, 's' to save word", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self, camera_id=0):
        """Run real-time detection"""
        print(f"🚀 Starting Real-Time ASL Detection")
        print(f"📱 Model: {self.model_type}")
        print(f"💻 Device: {self.device}")
        print(f"🎯 Confidence Threshold: {self.confidence_threshold}")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("✅ Camera opened successfully")
        print("\n📋 Instructions:")
        print("• Show your hand to the camera")
        print("• Make ASL fingerspelling gestures")
        print("• Press 'q' to quit")
        print("• Press 'c' to clear current word")
        print("• Press 's' to save current word")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract hand region
                hand_region, bbox = self.extract_hand_region(frame)
                
                # Predict sign
                prediction, confidence = self.predict_sign(hand_region)
                
                # Smooth predictions
                stable_prediction = self.smooth_predictions(prediction, confidence)
                
                # Update word buffer
                self.update_word_buffer(stable_prediction)
                
                # Draw UI
                self.draw_ui(frame, prediction, confidence, stable_prediction)
                
                # Show frame
                cv2.imshow('ASL Fingerspelling Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.word_buffer.clear()
                    print("🗑️ Word buffer cleared")
                elif key == ord('s'):
                    word = ''.join(self.word_buffer)
                    if word.strip():
                        print(f"💾 Saved word: '{word}'")
                        with open('asl_words.txt', 'a') as f:
                            f.write(word + '\n')
                    self.word_buffer.clear()
        
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Camera closed")

# =====================================
# MAIN FUNCTION
# =====================================

def main():
    """Main function to run real-time detection"""
    
    # Configuration
    MODEL_TYPE = 'cnn'  # Change to 'vit' if you have a ViT model
    MODEL_PATH = None   # Path to your trained model (e.g., 'asl_cnn_model.pth')
    CONFIDENCE_THRESHOLD = 0.6
    CAMERA_ID = 0
    
    print("🔥 ASL Real-Time Fingerspelling Detection")
    print("=" * 50)
    
    # Create detector
    detector = RealTimeASLDetector(
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Run detection
    detector.run(camera_id=CAMERA_ID)

if __name__ == "__main__":
    main()