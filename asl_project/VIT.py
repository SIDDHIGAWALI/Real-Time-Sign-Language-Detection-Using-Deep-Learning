import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Vision Transformer Implementation
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

class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, num_classes=30,
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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
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
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)

def load_vit_model(model_path, num_classes=29, img_size=64):
    """Load trained ViT model"""
    model = VisionTransformer(
        img_size=img_size,
        patch_size=8,
        num_classes=num_classes,
        embed_dim=128,
        depth=3,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_frame_vit(frame, image_size=64):
    """Preprocess frame for ViT"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

def run_vit_detection():
    """Run real-time ASL detection with Vision Transformer"""
    # Load model
    model_path = "models/asl_vit_model.pth"
    model = load_vit_model(model_path)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Vision Transformer ASL Detection Started. Press 'q' to quit.")
    
    # For prediction smoothing
    prediction_history = []
    history_size = 7  # Slightly larger for ViT
    confidence_threshold = 0.6  # Lower threshold for ViT
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create ROI for hand detection
        h, w = frame.shape[:2]
        roi_size = 280
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Preprocess and predict
        input_tensor = preprocess_frame_vit(roi)
        
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
        
        # Get most frequent prediction if confidence > threshold
        if confidence_score > confidence_threshold:
            most_common = max(set([p[0] for p in prediction_history]), 
                            key=[p[0] for p in prediction_history].count)
            avg_confidence = np.mean([p[1] for p in prediction_history if p[0] == most_common])
            display_text = f"Prediction: {most_common} ({avg_confidence:.2f})"
        else:
            display_text = "Low Confidence"
        
        # Draw ROI rectangle (blue for ViT)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Display prediction
        cv2.putText(frame, display_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Vision Transformer Model", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Place hand in blue box", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show top 3 predictions
        top_3 = torch.topk(probabilities, 3)
        for i, (prob, idx) in enumerate(zip(top_3.values[0], top_3.indices[0])):
            text = f"{i+1}. {ASL_CLASSES[idx]}: {prob:.2f}"
            cv2.putText(frame, text, (w-200, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('ASL Detection - Vision Transformer', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vit_detection()