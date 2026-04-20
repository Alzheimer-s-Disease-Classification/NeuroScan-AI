#NeuroScan AI: Multi-Stage Alzheimer's Detection

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/sheema216/NeuroScan-AI)
[![Python](https://img.shields.io/badge/Python-3.10-yellow)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-green)](https://flask.palletsprojects.com/)

NeuroScan AI is an Explainable Artificial Intelligence (XAI) web application designed to assist in the early detection and multi-stage classification of Alzheimer's Disease using MRI scans. 

Instead of operating as a "black box," this tool generates **Grad-CAM heatmaps** to visually highlight the specific regions of the brain that the model analyzed to make its prediction, providing transparency and trust for medical professionals.

##Live Demo
The application is fully containerized with Docker and deployed live on Hugging Face Spaces.
**Try it here:** [NeuroScan AI on Hugging Face](https://huggingface.co/spaces/sheema216/NeuroScan-AI)

##Technical Architecture
* **Deep Learning Model:** Fine-tuned `ResNet50` architecture.
* **Classification Stages:**
    * Mild Demented
    * Moderate Demented
    * Non-Demented
    * Very Mild Demented
* **Explainability (XAI):** Gradient-weighted Class Activation Mapping (Grad-CAM) integrated with OpenCV for brain-masking and overlay generation.
* **Backend:** Python & Flask.
* **Deployment:** Docker, Git Large File Storage (LFS), Hugging Face Spaces.

##Local Installation
To run this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SheemaWani/NeuroScan-AI.git](https://github.com/SheemaWani/NeuroScan-AI.git)
   cd NeuroScan-AI
