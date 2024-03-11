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
The project utilizes the **_publicly available HAM10000 dataset_** with below informations:
```
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
```
---
## Implementation
_First, clone this repository to your computer and then follow the instructions below._
### Step 1: Creat virtual/conda environment
- _With virtual environment_
```
python -m venv gemini-chatbot
source gemini-chatbot/bin/activate #for ubuntu
gemini-chatbot/Scripts/activate #for windows
```
- _With conda environment_
```
conda create --name gemini-chatbot
```
_After the installation, run command below to activate conda environment_
```
conda activate gemini-chatbot
```

### Step 2: Install libraries
```
cd ./Gemini-Medical-Chatbot
pip install -r requirements.txt
```

### Step 3: Run chatbot interface with Streamlit
```
streamlit run app.py
```
---

> [!NOTE]
> - This chatbot right now just answer to questions related to medical topic and in Vietnamese, this is because of ```keywords.txt```. Users can modify this with English keywords or any other languages.
> - To ensure that the model responds only within the scope of the medical field, I have created a file containing keywords related to the medical topic. Users can update ```keywords.txt``` file after cloning this repository to enhance the accuracy/flexibility of the chatbot.
