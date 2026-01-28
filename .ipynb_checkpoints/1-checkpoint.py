import torch
import torch.nn.functional as F
import cv2
import numpy as np

# ========== Define the model architectures you used for training ==========

class ResNetASL(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class GNNASL(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Replace this code with your actual GNN architecture
        pass
    def forward(self, x):
        # Replace with your actual forward logic
        return x

class ViTASL(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Replace this code with your actual ViT architecture
        pass
    def forward(self, x):
        # Replace with your actual forward logic
        return x

# ========== Initialize and Load Models ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 26  # A-Z fingerspelling, modify as needed

resnet_model = ResNetASL(num_classes).to(device)
resnet_model.load_state_dict(torch.load("models/asl_resnet_model.pth", map_location=device))
resnet_model.eval()

gnn_model = GNNASL(num_classes).to(device)
gnn_model.load_state_dict(torch.load("models/fast_asl_gnn_final.pth", map_location=device))
gnn_model.eval()

vit_model = ViTASL(num_classes).to(device)
vit_model.load_state_dict(torch.load("models/asl_vit_model.pth", map_location=device))
vit_model.eval()

# ========== Letter mapping (A-Z) ==========
labels = [chr(ord('A') + i) for i in range(26)]

# ========== Utility: Preprocessing ==========
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))  # Channels-first
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image, dtype=torch.float).unsqueeze(0).to(device)
    return image

# ========== Real-time Camera Detection ==========
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess(frame)

    with torch.no_grad():
        out_resnet = F.softmax(resnet_model(img), dim=1)
        out_gnn = F.softmax(gnn_model(img), dim=1)
        out_vit = F.softmax(vit_model(img), dim=1)

        # Average ensemble
        ensemble_out = (out_resnet + out_gnn + out_vit) / 3.0
        pred_idx = torch.argmax(ensemble_out, dim=1).item()
        pred_letter = labels[pred_idx]

    # Display prediction on frame
    cv2.putText(frame, f'ASL Pred: {pred_letter}', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("ASL Fingerspelling Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
