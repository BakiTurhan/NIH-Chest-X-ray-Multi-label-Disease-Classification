# Chest X-Ray Disease Detection Application

This application uses trained DenseNet121 and VGG19 models to detect diseases from chest X-ray images. Users can upload their own X-ray images and view the predictions of the selected model.

## Disease Classes

The application can detect the following disease classes:
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. No Finding

## Installation

1. Install the required libraries:
```
pip install -r requirements.txt
```

2. Place the model files (.pth) in the same directory as the application:
   - `densenet121_best_model.pth`
   - `vgg19_best_multilabel.pth`

## Running the Application

```
streamlit run app.py
```

## Usage

1. Select the model you want to use (DenseNet121 or VGG19) from the sidebar.
2. Upload a chest X-ray image in the "Upload chest X-ray" section.
3. You can view the model predictions for the uploaded image on the right side.

## Notes

- This application is for educational purposes only and does not replace real medical diagnosis.
- Model predictions cannot substitute for actual medical diagnoses.
- Always consult healthcare professionals. 