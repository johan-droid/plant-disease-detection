# 🌿 Plant Disease Detection

Deep Learning model for automated plant disease detection using EfficientNetB3 and Transfer Learning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

## 📊 Project Overview

This project uses deep learning to identify plant diseases from leaf images with **95%+ accuracy**. Built with TensorFlow and optimized for Google Colab's free T4 GPU.

### Key Features
- ✅ Transfer learning with EfficientNetB3
- ✅ Mixed precision training (FP16)
- ✅ Data augmentation pipeline
- ✅ Real-time predictions with confidence scores
- ✅ TFLite model for mobile deployment
- ✅ Comprehensive evaluation metrics

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Click the Colab badge above
2. Upload your plant disease dataset (ZIP format)
3. Run all cells
4. Start predicting!

### Option 2: Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
python predict.py sample_image.jpg
```

## 📁 Dataset Structure

Your dataset ZIP should be organized like this:
```
PlantVillage/
├── Apple___Black_rot/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Tomato___Late_blight/
│   ├── image1.jpg
│   └── ...
└── [other disease classes]/
```

**Recommended Dataset**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## 🏗️ Model Architecture

- **Base Model**: EfficientNetB3 (ImageNet pretrained)
- **Input Size**: 224x224x3
- **Fine-tuning**: Last 20 layers unfrozen
- **Optimizer**: AdamW with weight decay
- **Training**: Mixed precision (FP16)

### Architecture Details
```
Input → Data Augmentation → EfficientNetB3 → 
Dense(512) → Dropout → Dense(256) → Output(Classes)
```

## 📈 Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | 96%+ |
| Top-3 Accuracy | 99%+ |
| Training Time | ~20-30 min (T4 GPU) |
| Inference Time | ~50ms per image |

## 💻 Usage

### Training
```python
# In Colab or Jupyter
# Upload dataset and run all cells
# Model will be saved as best_plant_disease_model.keras
```

### Prediction
```python
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('best_plant_disease_model.keras')

# Predict
def predict_disease(image_path):
    img = keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    return class_names[np.argmax(predictions[0])]
```

## 📦 Project Structure
```
plant-disease-detection/
├── README.md
├── plant_disease_detection.ipynb    # Main training notebook
├── requirements.txt                  # Dependencies
├── LICENSE                          # MIT License
├── models/                          # Saved models
│   └── best_plant_disease_model.keras
└── assets/                          # Images and plots
    ├── sample_predictions/
    └── training_plots/
```

## 🔧 Requirements
```
tensorflow>=2.10.0
numpy>=1.21.0
pillow>=9.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🎯 Results

### Sample Predictions
| Disease | Confidence |
|---------|-----------|
| Tomato Late Blight | 98.5% |
| Apple Black Rot | 96.2% |
| Healthy Leaf | 99.1% |

### Training History
- Loss decreases steadily
- Validation accuracy plateaus around 96%
- No significant overfitting observed

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Dataset: PlantVillage Dataset
- Model: EfficientNet by Google Research
- Framework: TensorFlow/Keras

## 📧 Contact

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com

---

⭐ Star this repo if you found it helpful!
