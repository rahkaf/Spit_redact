# 🌱 Plant Disease Classification using MobileNetV2

### 🔍 Deep Learning Model for Detecting 38+ Crop Diseases

This project uses a **MobileNetV2** Convolutional Neural Network (CNN) to classify plant leaf diseases with high accuracy.
The model is lightweight, fast, and optimized for real-time inference on web apps and edge devices.

## 🚀 Live Demo

🔗 **Streamlit App:**
👉 [https://agriguard271005.streamlit.app/](https://agriguard271005.streamlit.app/)

## 🤖 Download / Use the Model

🔗 **Hugging Face Model:**
👉 [https://huggingface.co/Daksh159/plant-disease-mobilenetv2](https://huggingface.co/Daksh159/plant-disease-mobilenetv2)

You can load it using PyTorch:

```python
import torch
from torchvision import transforms
from PIL import Image

model = torch.load("plant_disease_mobilenetv2.pth", map_location="cpu")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open("your_image.jpg")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    prediction = output.argmax(dim=1).item()

print("Predicted class:", prediction)
```

## 🌿 Dataset

The model is trained on the **New Plant Diseases Dataset (Augmented)** containing **87,000+ images** across **38 classes**.

🔗 Dataset Link (Kaggle):
[https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## 🏗 Model Architecture

This model uses:

* **MobileNetV2 Backbone**
* Pretrained on **ImageNet**
* Final FC layer replaced for 38-class classification
* Data Augmentation
* EarlyStopping + ReduceLROnPlateau
* CrossEntropy Loss


## 📊 Training Setup

| Component     | Value        |
| ------------- | ------------ |
| Framework     | PyTorch      |
| Optimizer     | Adam         |
| Learning Rate | 1e-4         |
| Image Size    | 224 × 224    |
| Batch Size    | 32           |
| Epochs        | 25–40        |
| Loss Function | CrossEntropy |
| Base Model    | MobileNetV2  |

## 📈 Results

| Metric              | Value           |
| ------------------- | --------------- |
| Training Accuracy   | ~95%            |
| Validation Accuracy | ~94–95%         |
| Inference Speed     | <50ms per image |


## 🖼 Model Features

✔ Fast and lightweight

✔ Good for mobile & web apps

✔ Real-time detection

✔ Robust performance on unseen images

✔ Trained on augmented data


## ⚙ Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/yourusername/plant-disease-mobilenetv2.git
cd plant-disease-mobilenetv2
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app locally

```bash
streamlit run app.py
```


## 🧪 Testing the Model

Place your test image in the project folder and run:

```python
python predict.py --image path_to_image.jpg
```


## 📬 Contact

If you'd like to collaborate or improve the project, feel free to open an issue or reach out!


## 📄 License

This project is open-source under the **Apache License 2.0**.


Just tell me **“make aesthetic version"** or **“make minimal version"**!
