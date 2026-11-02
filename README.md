# Diabetes Risk Assessment System Using Machine Learning

## Project Overview

The **Diabetes Risk Assessment System** is a Python-based application that uses Machine Learning to predict whether a person's health measurements correspond to the diabetic or non-diabetic category.

The project uses a **Support Vector Machine (SVM)** classification algorithm and provides a graphical user interface (GUI) developed using Tkinter. Users can enter health-related measurements and view the model's prediction along with a visual representation of its score.

This project demonstrates the application of machine learning, data preprocessing, model evaluation, and graphical user interface development in healthcare-related data analysis.

> **Disclaimer:** This application is developed for educational purposes only. Its predictions are not a medical diagnosis and should not replace professional medical advice or laboratory testing.

## Objectives

- Develop a machine learning model for diabetes classification.
- Preprocess the dataset and handle missing numerical values.
- Train an SVM classifier using eight health-related features.
- Evaluate the model using standard classification metrics.
- Develop an interactive desktop application using Tkinter.
- Display predictions and a visual score gauge.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing, model training, and evaluation |
| Support Vector Machine (SVM) | Binary classification |
| Tkinter | Graphical user interface |
| Matplotlib | Score visualization |
| Joblib | Saving and loading the trained model |
| Visual Studio Code / Python IDLE | Writing and executing Python code |

## Dataset

The project uses a diabetes dataset containing the following input features:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. Body Mass Index (BMI)
7. Diabetes Pedigree Function
8. Age

The target column is `Outcome`, which represents the class label:

- `0` — Non-diabetic class
- `1` — Diabetic class

The dataset is divided into training and testing sets. The model is trained using the training data and evaluated on the test data.

## Machine Learning Algorithm

The project uses a **Support Vector Machine with an RBF kernel**.

The implementation includes:

- Median imputation for handling missing numerical values.
- StandardScaler for feature standardization.
- An SVM classifier for binary classification.
- An 80:20 training-testing split.
- Model evaluation using accuracy, precision, recall, confusion matrix, and classification report.

The trained model is saved using Joblib so that it can be loaded by the GUI without retraining every time the application starts.

## Project Structure

```text
Diabetes-Risk-Assessment-System/
│
├── app.py
├── train_model.py
├── diabetes.csv
├── model.pkl
├── README.md
└── requirements.txt
```

**Note:** This structure assumes that the training code is saved as `train_model.py`. If your files have different names, update this structure accordingly. Include only the files that actually exist in your repository.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip package manager
- Required Python libraries
- Dataset file: `diabetes.csv`

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Diabetes-Risk-Assessment-System.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Navigate to the Project Folder

```bash
cd Diabetes-Risk-Assessment-System
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

If the training script is named `train_model.py`, run:

```bash
python train_model.py
```

This trains the classifier, evaluates its performance, and saves the trained model as `model.pkl`.

If the trained model is already included in the repository, this step may be skipped.

### Step 5: Run the Application

```bash
python app.py
```

The graphical interface should open, allowing users to enter measurements and perform an assessment.

## Requirements

The `requirements.txt` file should contain:

```text
pandas
numpy
scikit-learn
matplotlib
joblib
```

Tkinter is generally included with standard Windows Python installations.

## Application Features

- User-friendly graphical interface.
- Eight input fields for health-related measurements.
- Input validation and error messages.
- SVM-based binary classification.
- Visual representation of the model's score.
- Assess Risk button for generating predictions.
- Reset button for clearing input fields.
- Medical disclaimer for responsible use.

## Model Evaluation

The model's performance can be evaluated using:

- **Accuracy:** Proportion of correctly classified observations.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **Confusion Matrix:** Summary of correct and incorrect classifications.
- **Classification Report:** Precision, recall, and F1-score for each class.

Actual evaluation results should be added after running the model. No performance figures are reported here because they have not been provided.

## Limitations

- Predictions depend on the quality and representativeness of the dataset.
- The model uses a limited number of input features.
- The system may not generalize to every population.
- The displayed score is not a clinically validated probability.
- The application is not intended for clinical diagnosis.

## Future Scope

- Evaluate additional machine learning algorithms.
- Improve model performance through hyperparameter tuning.
- Use larger and more diverse datasets.
- Develop a web or mobile application.
- Add secure patient history and report-generation features.
- Incorporate explainable AI techniques to improve prediction transparency.

## Learning Outcomes

- Understanding supervised machine learning and binary classification.
- Learning data preprocessing with Pandas and Scikit-learn.
- Training and evaluating an SVM classifier.
- Saving and loading trained models using Joblib.
- Developing a desktop GUI using Tkinter.
- Visualizing results using Matplotlib.

## Author

**Project:** Diabetes Risk Assessment System Using Machine Learning

**Language:** Python

**Project Type:** Machine Learning Mini Project

## Disclaimer

This project is intended solely for educational and demonstration purposes. It is not a medically validated diagnostic tool. Users should consult qualified healthcare professionals for diabetes screening, diagnosis, and treatment.