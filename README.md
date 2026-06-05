## Bangla Handwritten Word Recognition System

This project recognizes handwritten Bangla characters using a Convolutional Neural Network (CNN). Users can draw Bangla characters or words in a Streamlit application and receive predictions from the trained model. The project also includes MLflow experiment tracking and Docker support for deployment.

## Screenshots

| Streamlit Application | MLflow Tracking |
|----------------------|-----------------|
| ![Streamlit App](screenshots/streamlit_app.png) | ![MLflow](screenshots/mlflow_experiment.png) |

## Features

- Handwritten Bangla character recognition
- Character segmentation from user drawings
- Prediction confidence scores
- MLflow experiment tracking
- Streamlit web application
- Docker support
- Comparison of two CNN architectures

## Project Structure

```text
Bengali-Word-Predictor/
├── train.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── labels.json
├── github_link.txt
├── models/
│   └── model.keras
├── artifacts/
│   └── mlflow/
└── screenshots/
    ├── streamlit_app.png
    └── mlflow_experiment.png
```

## Dataset

The model was trained using the BanglaLekha-Isolated dataset. The dataset contains 84 classes of Bangla handwritten characters with approximately 166,000 images.

Dataset Link:

https://data.mendeley.com/datasets/hf6sf8zrkc/2

After downloading and extracting the dataset, the directory structure should look like:

```text
Bengali-Word-Predictor/
└── BanglaLekha-Isolated/
    └── Images/
        ├── 1/
        ├── 2/
        ├── 3/
        ...
        └── 84/
```

## Requirements

- Python 3.12
- pip
- Git
- Docker Desktop (optional)

## Installation

Clone the repository:

```bash
git clone https://github.com/Rajesh-Mojumder/Bengali-Word-Predictor.git
cd Bengali-Word-Predictor
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training

Run both model experiments:

```bash
python train.py --run_both
```

Run a single experiment:

```bash
python train.py --arch cnn --epochs 20 --lr 0.001
```

### Training Arguments

| Argument | Default Value | Description |
|----------|--------------|-------------|
| --arch | cnn | Model architecture |
| --lr | 0.001 | Learning rate |
| --epochs | 20 | Number of epochs |
| --batch_size | 64 | Batch size |
| --data_dir | BanglaLekha-Isolated/Images | Dataset path |
| --max_per_class | 300 | Maximum images per class |
| --run_both | None | Runs both architectures |

## MLflow Tracking

Launch the MLflow interface:

```bash
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open the following URL in your browser:

```text
http://127.0.0.1:5000
```

### Logged Information

- Training parameters
- Training and validation metrics
- Test performance metrics
- Trained model files
- Label mappings

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

### Usage

1. Draw a Bangla character or word on the canvas.
2. Click the recognition button.
3. View the predicted output.
4. Review confidence scores for each prediction.

## Docker Deployment

Build the Docker image:

```bash
docker build -t bangla-ocr-app:0.1 .
```

Run the container:

```bash
docker run -p 8501:8501 bangla-ocr-app:0.1
```

Open:

```text
http://localhost:8501
```

Ensure that the trained model file exists before creating the Docker image.

## Model Architectures

Two CNN architectures were implemented and compared during training.

### CNN

```text
Input (32x32x1)
  -> Conv2D(32) + BatchNorm + MaxPool
  -> Conv2D(64) + BatchNorm + MaxPool
  -> Conv2D(128) + BatchNorm + GlobalAvgPool
  -> Dense(256) + Dropout(0.4)
  -> Dense(84, Softmax)
```

### Deeper CNN

```text
Input (32x32x1)
  -> Conv2D(32) x2 + BatchNorm + MaxPool + Dropout(0.25)
  -> Conv2D(64) x2 + BatchNorm + MaxPool + Dropout(0.25)
  -> Conv2D(128) + BatchNorm + GlobalAvgPool
  -> Dense(512) + Dropout(0.5)
  -> Dense(84, Softmax)
```

Both architectures use the Adam optimizer and sparse categorical cross-entropy loss.

## Data Preprocessing

| Step | Description |
|--------|-------------|
| Color Conversion | Grayscale |
| Resize | 32 × 32 pixels |
| Normalization | Pixel values scaled to 0–1 |
| Augmentation | Rotation, shift, and zoom |
| Data Split | 75% train, 15% validation, 10% test |

## Character Segmentation

The handwritten input is processed using the following steps:

1. Convert the image to grayscale.
2. Apply thresholding to separate foreground and background.
3. Remove noise using morphological operations.
4. Detect connected components.
5. Filter small components.
6. Sort components from left to right.
7. Crop and prepare each component for prediction.

## Command Reference

| Task | Command |
|--------|----------|
| Train both models | `python train.py --run_both` |
| Train one model | `python train.py --arch cnn` |
| Start MLflow | `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` |
| Run Streamlit | `streamlit run app.py` |
| Build Docker image | `docker build -t bangla-ocr-app:0.1 .` |
| Run Docker container | `docker run -p 8501:8501 bangla-ocr-app:0.1` |

## Limitations

- Performance decreases when characters overlap.
- Complex Bangla ligatures may not be recognized accurately.
- The system depends on character-level segmentation.
- No language model is used during prediction.
- Recognition quality depends on handwriting clarity.

## Future Improvements

- Implement ResNet or EfficientNet architectures.
- Add transfer learning approaches.
- Integrate sequence-based recognition models such as BiLSTM with CTC.
- Use a Bangla language model for post-processing.
- Train on the complete dataset using GPU resources.
- Increase input image resolution.
- Deploy on cloud platforms for public access.

## Author

Rajesh Mojumder

SICIP, BRAC University

Course Project