# 🐶🐱 Dog vs. Cat Image Classification with Transfer Learning

This project focuses on fine-tuning multiple pre-trained deep learning models to classify images of dogs and cats using the [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats). It leverages **TensorFlow**, **Keras**, and **Transfer Learning** to efficiently adapt powerful models to a binary classification task.

## 🚀 Models Used

The following pre-trained models were fine-tuned:

* **ResNet50**
* **VGG16**
* **MobileNetV2**
* **EfficientNetB0**
* **YOLO** 

Each model was adapted by:

* Removing the top classification layers
* Adding custom dense layers
* Freezing and unfreezing layers strategically
* Using `ImageDataGenerator` for real-time data augmentation

## 🧠 Approach

* **Transfer Learning:** Reused pre-trained weights (from ImageNet) to reduce training time and improve performance on a small dataset.
* **Data Augmentation:** Applied random flips, zoom, and rotations to generalize better.
* **Evaluation Metrics:** Accuracy, precision, recall, and F1-score.
* **YOLO:** Applied separately for object detection on dog/cat images.

## 🛠️ Tech Stack

* Python 3.x
* TensorFlow / Keras
* OpenCV (for image processing)
* NumPy, Matplotlib
* YOLOv4 / YOLOv5 (via pre-trained weights or TensorFlow implementation)

## 📊 Training Configuration

* **Loss Function:** `binary_crossentropy`
* **Optimizer:** `Adam`
* **Epochs:** 10–20 per model
* **Batch Size:** 32
* **Input Size:** Resized to 224x224 (or 416x416 for YOLO)

## 📈 Results

| Model         | Precision | Recall  |
| :------------ | :-------- | :------ |
| ResNet50      | 0.9863    | 0.9951  |
| VGG16         | 0.9872    | 0.9921  |
| MobileNetV2   | 0.9814    | 0.9901  |
| EfficientNetB0| 0.9881    | 0.9881  |
| YOLOV11       | 0.9832    | 0.9812  |

*(Exact metrics depend on training duration and data split)*

## 🔍 Sample Predictions

* Classification: Model outputs probabilities of "dog" vs. "cat".
* Detection (YOLO): Bounding boxes over dog/cat with confidence scores.

## 📂 Folder Structure

```
├── 4 models cats dogs.ipynb
├── /data
│   ├── /train
│   └── /validation
├── /models
├── /output
└── README.md
```

## 📝 How to Run

```bash
pip install tensorflow numpy matplotlib opencv-python
```

Run the notebook:

```bash
jupyter notebook "4 models cats dogs.ipynb"
```

Or adapt into a Python script and run training/evaluation in your environment.

---

## 📌 Notes

* You can experiment with different input image sizes for each model.
* Fine-tuning vs. feature extraction can be toggled via layer freezing.
* YOLO can be trained separately or used as an off-the-shelf detector for localization tasks.

## 📜 License

This project is open-source and available under the MIT License.

---

Let me know if you want to include visualizations, sample outputs, or deploy this as a web app!
