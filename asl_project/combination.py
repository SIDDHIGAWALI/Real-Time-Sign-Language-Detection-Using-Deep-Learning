import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from PIL import Image

# For GNN (requires torch-geometric)
try:
    from torch_geometric.data import Data as GeoData
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm
    GNN_AVAILABLE = True
except ImportError:
    print("Warning: torch-geometric not installed. GNN model will be disabled.")
    GNN_AVAILABLE = False

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

####################################
# ResNet18 Model Functions
####################################
def create_resnet_model(num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_resnet_model(model_path, num_classes=29):
    model = create_resnet_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_frame_resnet(frame, image_size=224):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_image).unsqueeze(0)
    return tensor


####################################
# Vision Transformer (ViT) Model
####################################
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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_image).unsqueeze(0)
    return tensor


####################################
# Graph Neural Network (GNN) Model
####################################
if GNN_AVAILABLE:
    class FastASLGNN(nn.Module):
        def __init__(self, input_dim=6, hidden_dim=64, num_classes=29, dropout=0.1):
            super().__init__()
            self.input_transform = nn.Linear(input_dim, hidden_dim)
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.bn1 = BatchNorm(hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.bn2 = BatchNorm(hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.bn3 = BatchNorm(hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
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
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_global = torch.cat([x_mean, x_max], dim=1)
            return self.classifier(x_global)

    def extract_patches_and_graph(img_tensor, patch_size=4):
        C, H, W = img_tensor.shape
        patches_per_dim = H // patch_size
        
        patches = []
        positions = []
        for i in range(patches_per_dim):
            for j in range(patches_per_dim):
                patch = img_tensor[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patch_flat = patch.flatten()
                features = [
                    patch_flat.mean().item(),
                    patch_flat.std().item() + 1e-8,
                    patch_flat.max().item(),
                    patch_flat.min().item(),
                    i / max(1, patches_per_dim - 1),
                    j / max(1, patches_per_dim - 1),
                ]
                patches.append(features)
                positions.append((i, j))
        
        edges = []
        pos_dict = {pos: idx for idx, pos in enumerate(positions)}
        for idx, (i, j) in enumerate(positions):
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if (ni, nj) in pos_dict:
                    edges.append([idx, pos_dict[(ni,nj)]])
        if not edges:
            edges = [[i, (i+1) % len(patches)] for i in range(len(patches))]
        edges = edges + [[e[1], e[0]] for e in edges]
        
        return torch.tensor(patches, dtype=torch.float32), torch.tensor(edges, dtype=torch.long).t().contiguous()

    def load_gnn_model(model_path, input_dim=6, hidden_dim=64, num_classes=29):
        model = FastASLGNN(input_dim, hidden_dim, num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    def preprocess_frame_gnn(frame, image_size=32):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        tensor = transform(pil_image)
        return tensor

    def create_graph_data(img_tensor):
        node_features, edge_index = extract_patches_and_graph(img_tensor)
        data = GeoData(
            x=node_features,
            edge_index=edge_index,
            batch=torch.zeros(node_features.size(0), dtype=torch.long)
        )
        return data

####################################
# Real-time Multi-Model ASL Detection
####################################
def run_combined_detection():
    # Load models paths
    path_resnet = "models/asl_resnet_model.pth"
    path_vit = "models/asl_vit_model.pth"
    path_gnn = "models/fast_asl_gnn_final.pth"

    # Load models
    model_resnet = load_resnet_model(path_resnet)
    model_vit = load_vit_model(path_vit)

    if GNN_AVAILABLE:
        try:
            model_gnn = load_gnn_model(path_gnn)
        except Exception as e:
            print(f"GNN model loading error: {e}")
            model_gnn = None
    else:
        model_gnn = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Combined ASL Detection Started. Press 'q' to quit.")

    prediction_history_resnet = []
    prediction_history_vit = []
    prediction_history_gnn = []

    history_size_resnet = 5
    history_size_vit = 7
    history_size_gnn = 5

    confidence_thresh_resnet = 0.7
    confidence_thresh_vit = 0.6
    confidence_thresh_gnn = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        # Define ROI differently for each model to accommodate their input sizes
        # Center ROI is generally fine.

        # ResNet ROI (300x300)
        roi_size_resnet = 300
        x1_r = (w - roi_size_resnet) // 2
        y1_r = (h - roi_size_resnet) // 2
        x2_r = x1_r + roi_size_resnet
        y2_r = y1_r + roi_size_resnet
        roi_resnet = frame[y1_r:y2_r, x1_r:x2_r]

        # ViT ROI (280x280)
        roi_size_vit = 280 
        x1_v = (w - roi_size_vit) // 2
        y1_v = (h - roi_size_vit) // 2
        x2_v = x1_v + roi_size_vit
        y2_v = y1_v + roi_size_vit
        roi_vit = frame[y1_v:y2_v, x1_v:x2_v]

        # GNN ROI (250x250) if enabled
        roi_size_gnn = 250
        x1_g = (w - roi_size_gnn) // 2
        y1_g = (h - roi_size_gnn) // 2
        x2_g = x1_g + roi_size_gnn
        y2_g = y1_g + roi_size_gnn
        roi_gnn = frame[y1_g:y2_g, x1_g:x2_g]

        # Process ResNet
        input_resnet = preprocess_frame_resnet(roi_resnet)
        with torch.no_grad():
            out_resnet = model_resnet(input_resnet)
            probs_resnet = torch.softmax(out_resnet, dim=1)
            conf_resnet, pred_resnet = torch.max(probs_resnet, 1)
            pred_class_resnet = ASL_CLASSES[pred_resnet.item()]
            conf_score_resnet = conf_resnet.item()

        prediction_history_resnet.append((pred_class_resnet, conf_score_resnet))
        if len(prediction_history_resnet) > history_size_resnet:
            prediction_history_resnet.pop(0)
        if conf_score_resnet > confidence_thresh_resnet:
            most_common_r = max(set([p[0] for p in prediction_history_resnet]),
                                key=[p[0] for p in prediction_history_resnet].count)
            avg_conf_r = np.mean([p[1] for p in prediction_history_resnet if p[0] == most_common_r])
            display_resnet = f"ResNet: {most_common_r} ({avg_conf_r:.2f})"
        else:
            display_resnet = "ResNet: Low Confidence"

        # Process ViT
        input_vit = preprocess_frame_vit(roi_vit)
        with torch.no_grad():
            out_vit = model_vit(input_vit)
            probs_vit = torch.softmax(out_vit, dim=1)
            conf_vit, pred_vit = torch.max(probs_vit, 1)
            pred_class_vit = ASL_CLASSES[pred_vit.item()]
            conf_score_vit = conf_vit.item()

        prediction_history_vit.append((pred_class_vit, conf_score_vit))
        if len(prediction_history_vit) > history_size_vit:
            prediction_history_vit.pop(0)
        if conf_score_vit > confidence_thresh_vit:
            most_common_v = max(set([p[0] for p in prediction_history_vit]),
                                key=[p[0] for p in prediction_history_vit].count)
            avg_conf_v = np.mean([p[1] for p in prediction_history_vit if p[0] == most_common_v])
            display_vit = f"ViT: {most_common_v} ({avg_conf_v:.2f})"
        else:
            display_vit = "ViT: Low Confidence"

        # Process GNN if available and model loaded
        if model_gnn:
            try:
                input_gnn_img = preprocess_frame_gnn(roi_gnn)
                graph_data = create_graph_data(input_gnn_img)
                with torch.no_grad():
                    out_gnn = model_gnn(graph_data)
                    probs_gnn = torch.softmax(out_gnn, dim=1)
                    conf_gnn, pred_gnn = torch.max(probs_gnn, 1)
                    pred_class_gnn = ASL_CLASSES[pred_gnn.item()]
                    conf_score_gnn = conf_gnn.item()
            except Exception as e:
                pred_class_gnn = "Error"
                conf_score_gnn = 0.0
                probs_gnn = torch.zeros(1, len(ASL_CLASSES))

            prediction_history_gnn.append((pred_class_gnn, conf_score_gnn))
            if len(prediction_history_gnn) > history_size_gnn:
                prediction_history_gnn.pop(0)
            if conf_score_gnn > confidence_thresh_gnn and pred_class_gnn != "Error":
                most_common_g = max(set([p[0] for p in prediction_history_gnn]),
                                    key=[p[0] for p in prediction_history_gnn].count)
                avg_conf_g = np.mean([p[1] for p in prediction_history_gnn if p[0] == most_common_g])
                display_gnn = f"GNN: {most_common_g} ({avg_conf_g:.2f})"
            else:
                display_gnn = "GNN: Low Confidence or Error"
        else:
            display_gnn = "GNN: Disabled"

        # Draw ROI boxes with colors for each model
        cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), (0, 255, 0), 2)   # Green for ResNet
        cv2.rectangle(frame, (x1_v, y1_v), (x2_v, y2_v), (255, 0, 0), 2)   # Blue for ViT
        if model_gnn:
            cv2.rectangle(frame, (x1_g, y1_g), (x2_g, y2_g), (0, 255, 255), 2) # Yellow for GNN

        # Display predictions on the frame
        cv2.putText(frame, display_resnet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, display_vit, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, display_gnn, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "Press 'q' to quit", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the frame
        cv2.imshow('ASL Detection - Combined Models', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_combined_detection()
