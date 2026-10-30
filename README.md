<div align="center">

# 🎨 Human vs AI Artwork Classification

**Can a neural network tell who painted it — a human artist or an AI model?**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

</div>

---

## 📌 Overview

AI image generators can now imitate famous art styles. This project trains an image classifier that recognizes **20 classes of artwork** — combining both the **art style** and **who created it**: a human artist or an AI model (Stable Diffusion, Latent Diffusion, DiffusionDB).

The final **ResNet50** transfer learning model reaches **81.6% accuracy on the test set**.

## 📊 Dataset

| | |
|---|---|
| **Source** | Kaggle — *Human & AI Artwork* (`human-ai-artwork`) |
| **Original size** | 52 classes |
| **Used in this project** | 20 selected classes, balanced to max. 350 training images per class |
| **Split** | train / validation / test |
| **Input size** | 224 × 224 RGB |

### Example classes

| 🧑‍🎨 Human-made | 🤖 AI-generated |
|---|---|
| `Human_Pop_Art` | `AI_SD_expressionism` (Stable Diffusion) |
| `Human_Post_Impressionism` | `AI_SD_realism` |
| `Human_Analytical_Cubism` | `AI_SD_surrealism` |
| `Human_Synthetic_Cubism` | `AI_SD_ukiyo-e` |
| `Human_Color_Field_Painting` | `AI_LD_impressionism` (Latent Diffusion) |
| `Human_Action_Painting` | `AI_LD_expressionism` |
| `Human_Symbolism` | `AI_DiffusionDB_large_3` |

## 🧠 Models

| # | Model | Details | Accuracy |
|---|---|---|:---:|
| 1 | Custom CNN | `Conv2D` + `MaxPool2D` blocks → dense layers · `EarlyStopping` | ~32% (val) |
| 2 | ResNet50 — frozen base | ImageNet weights, new softmax head · exponential LR decay | 80.7% (test) |
| 3 | **ResNet50 — second training round** ✅ | Continued training of the classifier head | **81.6% (test)** |

### 💡 Key insights

- 20 visually similar art styles are a hard problem — a CNN trained from scratch reaches only ~32%.
- **Transfer learning more than doubles the accuracy** (32% → 81.6%), because ResNet50 already knows textures, edges and shapes from ImageNet.
- The training accuracy (~98%) is higher than the test accuracy (~82%), so there is still room to reduce overfitting.

## 🚀 How to run

This notebook was written for **Kaggle Notebooks**:

1. Create a new Kaggle notebook and upload `ai-artwork.ipynb`.
2. Add the **human-ai-artwork** dataset from the right-hand panel.
3. Enable a **GPU** accelerator and run all cells.

## 📁 Project structure

```
human-ai-artwork-classification/
├── ai-artwork.ipynb   # dataset split, models, evaluation, prediction on new images
└── README.md
```

## 🔮 Future improvements

- Unfreeze the top ResNet50 layers and fine-tune with a small learning rate
- Stronger data augmentation and regularization to reduce overfitting
- A simpler second task: binary **Human vs AI** classification
- Try newer architectures (EfficientNetV2, ConvNeXt, Vision Transformer)
- Grad-CAM visualizations to show *which parts of the image* the model looks at

## 👤 Author

**Mahammad Hajiyev** · [@mahammadhajiiyev](https://github.com/mahammadhajiiyev)
