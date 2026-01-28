import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def create_resnet_model(num_classes):
    """Create ResNet18 model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_resnet_model(model_path, num_classes=29):
    """Load trained ResNet model"""
    model = create_resnet_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_frame(frame, image_size=224):
    """Preprocess frame for ResNet"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

def run_resnet_detection():
    """Run real-time ASL detection with ResNet"""
    # Load model
    model_path = "models/asl_resnet_model.pth"
    model = load_resnet_model(model_path)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("ResNet ASL Detection Started. Press 'q' to quit.")
    
    # For prediction smoothing
    prediction_history = []
    history_size = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create ROI for hand detection
        h, w = frame.shape[:2]
        roi_size = 300
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Preprocess and predict
        input_tensor = preprocess_frame(roi)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = ASL_CLASSES[predicted.item()]
            confidence_score = confidence.item()
        
        # Smooth predictions
        prediction_history.append((predicted_class, confidence_score))
        if len(prediction_history) > history_size:
            prediction_history.pop(0)
        
        # Get most frequent prediction if confidence > 0.7
        if confidence_score > 0.7:
            most_common = max(set([p[0] for p in prediction_history]), 
                            key=[p[0] for p in prediction_history].count)
            display_text = f"Prediction: {most_common} ({confidence_score:.2f})"
        else:
            display_text = "Low Confidence"
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display prediction
        cv2.putText(frame, display_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "ResNet Model", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Place hand in green box", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('ASL Detection - ResNet', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_resnet_detection()