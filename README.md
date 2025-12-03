# Real-Time-Sign-Language-Detection-Using-Deep-Learning

This project presents an advanced Sign Language Detection system that integrates multiple deep-learning architectures for highly accurate gesture recognition. The model combines:<br>
ResNet-CNN for extracting strong spatial and visual features from images.<br>
Vision Transformers (ViTs) to capture global context and long-range dependencies across frames.<br>
Graph Neural Networks (GNNs) for modeling the structural relationships between hand landmarks.<br>
By leveraging the strengths of these three architectures, the system provides robust, efficient, and real-time sign language classification. This approach enhances recognition accuracy, even in challenging conditions such as different lighting, backgrounds, and hand orientations.

✨ Features
Multi-model hybrid architecture (GNN + ViT + ResNet-CNN)<br>
High-accuracy sign language gesture classification<br>
Real-time gesture detection support<br>
Preprocessing pipeline for hand landmark extraction<br>
Modular and extensible codebase<br>

🧠 Technologies & Models Used
ResNet-50 / ResNet-18<br>
Vision Transformers (ViT-B/16)<br>
Graph Neural Networks (GCN / GAT)
<br>MediaPipe / OpenCV (for hand landmarks)
<br>PyTorch / TensorFlow

📁 Dataset
The project uses image/landmark-based datasets containing various sign language gestures. Custom datasets can also be added easily.

🚀 Future Enhancements
Deploy as a web or mobile application

Expand to full sentence-level sign recognition

Add support for more sign languages
