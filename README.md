# Real-Time-Sign-Language-Detection-Using-Deep-Learning

This project presents an advanced Sign Language Detection system that integrates multiple deep-learning architectures for highly accurate gesture recognition. The model combines:

ResNet-CNN for extracting strong spatial and visual features from images.

Vision Transformers (ViTs) to capture global context and long-range dependencies across frames.

Graph Neural Networks (GNNs) for modeling the structural relationships between hand landmarks.

By leveraging the strengths of these three architectures, the system provides robust, efficient, and real-time sign language classification. This approach enhances recognition accuracy, even in challenging conditions such as different lighting, backgrounds, and hand orientations.

✨ Features
Multi-model hybrid architecture (GNN + ViT + ResNet-CNN)
High-accuracy sign language gesture classification
Real-time gesture detection support
Preprocessing pipeline for hand landmark extraction
Modular and extensible codebase

🧠 Technologies & Models Used
ResNet-50 / ResNet-18
Vision Transformers (ViT-B/16)
Graph Neural Networks (GCN / GAT)
MediaPipe / OpenCV (for hand landmarks)
PyTorch / TensorFlow

📁 Dataset
The project uses image/landmark-based datasets containing various sign language gestures. Custom datasets can also be added easily.

🚀 Future Enhancements
Deploy as a web or mobile application

Expand to full sentence-level sign recognition

Add support for more sign languages
