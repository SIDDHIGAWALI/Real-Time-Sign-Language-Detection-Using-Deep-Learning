#!/usr/bin/env python3
"""
Real-time ASL Detection System
Supports multiple model architectures: GNN, ViT, and ResNet
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import argparse
import time
from pathlib import Path
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

class ASLDetector:
    def __init__(self, model_path, model_type='gnn', confidence_threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type.lower()
        self.confidence_threshold = confidence_threshold
        
        # ASL alphabet classes (adjust according to your training)
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Initialize MediaPipe for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Image transforms for CNN models
        if model_type in ['vit', 'resnet']:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def load_model(self, model_path):
        """Load the appropriate model based on model_type"""
        if self.model_type == 'gnn':
            model = self.create_gnn_model()
        elif self.model_type == 'vit':
            model = self.create_vit_model()
        elif self.model_type == 'resnet':
            model = self.create_resnet_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def create_gnn_model(self):
        """Create GNN model architecture - adjust according to your implementation"""
        # This is a placeholder - replace with your actual GNN architecture
        class SimpleGNN(nn.Module):
            def __init__(self, num_classes=26):
                super().__init__()
                # Assuming hand landmarks input (21 landmarks * 2 coordinates = 42 features)
                self.fc1 = nn.Linear(42, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_classes)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return SimpleGNN(len(self.classes))
    
    def create_vit_model(self):
        """Create Vision Transformer model"""
        from torchvision.models import vit_b_16
        model = vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(self.classes))
        return model
    
    def create_resnet_model(self):
        """Create ResNet model"""
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        return model
    
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]  # Get first hand
            # Convert landmarks to numpy array
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y])
            return np.array(coords, dtype=np.float32), landmarks
        
        return None, None
    
    def crop_hand_region(self, frame, landmarks):
        """Crop hand region from frame for CNN models"""
        if landmarks is None:
            return None
        
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        # Add padding around hand
        padding = 50
        x_min = max(0, int(min(x_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_min = max(0, int(min(y_coords)) - padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        # Crop hand region
        hand_crop = frame[y_min:y_max, x_min:x_max]
        
        if hand_crop.size == 0:
            return None
        
        return hand_crop
    
    def predict(self, frame):
        """Make prediction on frame"""
        # Extract hand landmarks
        landmarks_array, landmarks = self.extract_hand_landmarks(frame)
        
        if landmarks_array is None:
            return None, 0.0, None
        
        with torch.no_grad():
            if self.model_type == 'gnn':
                # Use landmarks directly for GNN
                input_tensor = torch.FloatTensor(landmarks_array).unsqueeze(0).to(self.device)
            else:
                # Use cropped hand image for CNN models
                hand_crop = self.crop_hand_region(frame, landmarks)
                if hand_crop is None:
                    return None, 0.0, landmarks
                
                input_tensor = self.transform(hand_crop).unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.classes[predicted.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score, landmarks
    
    def run_realtime_detection(self, source=0, save_video=False, output_path='asl_detection_output.mp4'):
        """Run real-time ASL detection"""
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Video writer setup if saving
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Performance tracking
        fps_counter = 0
        start_time = time.time()
        
        print(f"Starting ASL detection using {self.model_type.upper()} model...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror frame for better user experience
                frame = cv2.flip(frame, 1)
                
                # Make prediction
                prediction, confidence, landmarks = self.predict(frame)
                
                # Draw hand landmarks
                if landmarks:
                    self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Display prediction
                if prediction and confidence > self.confidence_threshold:
                    text = f"{prediction} ({confidence:.2f})"
                    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "No confident prediction", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update every 30 frames
                    elapsed = time.time() - start_time
                    fps = fps_counter / elapsed
                    fps_counter = 0
                    start_time = time.time()
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Model info
                cv2.putText(frame, f"Model: {self.model_type.upper()}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Save video frame
                if save_video:
                    out.write(frame)
                
                # Display frame
                cv2.imshow('ASL Real-time Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    screenshot_path = f'asl_screenshot_{timestamp}.jpg'
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            print("ASL detection session ended")

def main():
    parser = argparse.ArgumentParser(description='Real-time ASL Detection')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--model_type', type=str, choices=['gnn', 'vit', 'resnet'],
                       default='gnn', help='Type of model to use')
    parser.add_argument('--source', type=int, default=0,
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold for predictions')
    parser.add_argument('--save_video', action='store_true',
                       help='Save detection video to file')
    parser.add_argument('--output_path', type=str, default='asl_detection_output.mp4',
                       help='Output video path (if saving)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Initialize detector
    detector = ASLDetector(
        model_path=args.model_path,
        model_type=args.model_type,
        confidence_threshold=args.confidence
    )
    
    # Run detection
    detector.run_realtime_detection(
        source=args.source,
        save_video=args.save_video,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()