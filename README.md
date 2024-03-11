# Skin Disease Prediction app

## Overview 

The goal of this project is to develop a skin disease prediction application that utilizes machine learning techniques and AI models to predict the likelihood of various skin diseases based on input images. _**The application aims to assist users in identifying potential skin conditions by providing the top three probable diseases along with their accuracies.**_

---
## Features
- _**Image Upload**_: Users can upload images of skin lesions through the application interface.
- _**Prediction**_: Upon uploading an image, the application utilizes the trained model to predict the top three potential skin diseases along with their corresponding accuracies.
- _**User Interface**_: The application offers an intuitive user interface for seamless interaction and easy interpretation of results.

---
## Dataset
The project utilizes the _publicly available HAM10000 dataset_ with below informations:

- **Name**: HAM10000 (Human Against Machine with 10,000 training images)
- **Content**: Dermatoscopic images of various pigmented skin lesions
- **Size**: Contains 10,015 images
- **7 Classes**: Includes seven different diagnostic categories:
  - nv: Melanocytic nevus (common mole)
  - mel: Melanoma
  - bkl: Benign keratosis-like lesion
  - bcc: Basal cell carcinoma
  - akiec: Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
  - vasc: Vascular lesion
  - df: Dermatofibroma
- **Annotations**: Each image is accompanied by clinical metadata including patient age, sex, anatomical site of the lesion, and whether the lesion was malignant or benign.
For more informations and to download the dataset, users can reach to [_**Skin Cancer MNIST HAM10000.**_](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
---
## Model Architecture
In this project, _we leverage both MobileNet and ResNet architectures for training_. MobileNet's lightweight design accelerates training and reduces costs, while ResNet's depth enhances the model's ability to capture complex features. This dual approach optimizes both speed and accuracy, ensuring an efficient skin disease prediction system.

---

## Implementation

For model training, especially with a large dataset like ours, CPU/GPU resources are essential due to high RAM requirements. Therefore, I strongly advise users to conduct training within a virtual environment or a Conda environment to efficiently manage GPU resources.

First, clone this repository to your computer and then follow the instructions below.
### Step 1: Creat virtual/conda environment
- _With virtual environment_
```
python -m venv skin-disease
source skin-disease/bin/activate #for ubuntu
skin-diseas/Scripts/activate #for windows
```
- _With conda environment_
> [!NOTE]
> To utilize a Conda environment for training, users are required to install Anaconda or Miniconda. Further instructions and tutorials can be found at the following link: [Miniconda](https://docs.anaconda.com/free/miniconda/index.html).
```
conda create --name skin-disease
```
_After the installation, run command below to activate conda environment_
```
conda activate skin-disease
```

### Step 2: Install libraries
```
cd ./Skin-Diseases-Prediction
pip install -r requirements.txt
```

### Step 3: Train the model
```
python train.py    // Users need to check the source code file and change the path link to their dataset.
```

### Step 4: Run the app interface with trained model
```
python app.py    // Change the checkpoint name/path (right now is model.h5) in app.py file 
```
