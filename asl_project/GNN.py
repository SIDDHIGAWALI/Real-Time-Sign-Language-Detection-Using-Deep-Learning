import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

# For GNN (requires torch-geometric)
try:
    from torch_geometric.data import Data as GeoData
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm
    GNN_AVAILABLE = True
except ImportError:
    print("Warning: torch-geometric not installed. Install with: pip install torch-geometric")
    GNN_AVAILABLE = False

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Fast GNN Model Implementation
class FastASLGNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_classes=29, dropout=0.1):
        super().__init__()
        
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        
        # Simple GCN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.input_transform(x))
        
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_global = torch.cat([x_mean, x_max], dim=1)
        
        return self.classifier(x_global)

def extract_patches_and_graph(img_tensor, patch_size=4):
    """Extract patches and create graph structure"""
    C, H, W = img_tensor.shape
    patches_per_dim = H // patch_size
    
    patches = []
    positions = []
    
    for i in range(patches_per_dim):
        for j in range(patches_per_dim):
            # Extract patch
            patch = img_tensor[:, i*patch_size:(i+1)*patch_size, 
                              j*patch_size:(j+1)*patch_size]
            
            # Simple features
            patch_flat = patch.flatten()
            
            features = [
                patch_flat.mean().item(),           # Mean
                patch_flat.std().item() + 1e-8,    # Std
                patch_flat.max().item(),            # Max
                patch_flat.min().item(),            # Min
                i / max(1, patches_per_dim-1),      # Position X
                j / max(1, patches_per_dim-1),      # Position Y
            ]
            
            patches.append(features)
            positions.append((i, j))
    
    # Create edges (4-connectivity)
    edges = []
    pos_dict = {pos: idx for idx, pos in enumerate(positions)}
    
    for idx, (i, j) in enumerate(positions):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in pos_dict:
                edges.append([idx, pos_dict[(ni, nj)]])
    
    if not edges:
        edges = [[i, (i+1) % len(patches)] for i in range(len(patches))]
    
    # Make edges undirected
    edges = edges + [[e[1], e[0]] for e in edges]
    
    return torch.tensor(patches, dtype=torch.float32), torch.tensor(edges, dtype=torch.long).t().contiguous()

def load_gnn_model(model_path, input_dim=6, hidden_dim=64, num_classes=29):
    """Load trained GNN model"""
    if not GNN_AVAILABLE:
        raise ImportError("torch-geometric is required for GNN model")
    
    model = FastASLGNN(input_dim, hidden_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_frame_gnn(frame, image_size=32):
    """Preprocess frame for GNN"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    tensor = transform(pil_image)
    return tensor

def create_graph_data(img_tensor):
    """Create graph data from image tensor"""
    node_features, edge_index = extract_patches_and_graph(img_tensor)
    
    data = GeoData(
        x=node_features,
        edge_index=edge_index,
        batch=torch.zeros(node_features.size(0), dtype=torch.long)  # Single graph
    )
    
    return data

def run_gnn_detection():
    """Run real-time ASL detection with GNN"""
    if not GNN_AVAILABLE:
        print("Error: torch-geometric not installed. Please install it first:")
        print("pip install torch-geometric")
        return
    
    # Load model
    model_path = "models/fast_asl_gnn_final.pth"
    try:
        model = load_gnn_model(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Make sure you have trained and saved the GNN model.")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("GNN ASL Detection Started. Press 'q' to quit.")
    
    # For prediction smoothing
    prediction_history = []
    history_size = 5
    confidence_threshold = 0.5  # Lower threshold for GNN
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create ROI for hand detection
        h, w = frame.shape[:2]
        roi_size = 250
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        try:
            # Preprocess and create graph
            img_tensor = preprocess_frame_gnn(roi)
            graph_data = create_graph_data(img_tensor)
            
            # Predict
            with torch.no_grad():
                outputs = model(graph_data)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = ASL_CLASSES[predicted.item()]
                confidence_score = confidence.item()
        
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_class = "Error"
            confidence_score = 0.0
        
        # Smooth predictions
        prediction_history.append((predicted_class, confidence_score))
        if len(prediction_history) > history_size:
            prediction_history.pop(0)
        
        # Get most frequent prediction if confidence > threshold
        if confidence_score > confidence_threshold and predicted_class != "Error":
            most_common = max(set([p[0] for p in prediction_history]), 
                            key=[p[0] for p in prediction_history].count)
            avg_confidence = np.mean([p[1] for p in prediction_history if p[0] == most_common])
            display_text = f"Prediction: {most_common} ({avg_confidence:.2f})"
            text_color = (0, 255, 255)  # Yellow
        else:
            display_text = "Low Confidence" if predicted_class != "Error" else "Processing Error"
            text_color = (0, 0, 255)  # Red
        
        # Draw ROI rectangle (yellow for GNN)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Display prediction
        cv2.putText(frame, display_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, "Graph Neural Network Model", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Place hand in yellow box", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show graph info
        try:
            num_nodes = graph_data.x.size(0)
            num_edges = graph_data.edge_index.size(1)
            cv2.putText(frame, f"Nodes: {num_nodes}, Edges: {num_edges}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        except:
            pass
        
        cv2.imshow('ASL Detection - Graph Neural Network', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gnn_detection()