# Chest X-Ray Disease Detection Application  

**Prepared by:** Baki Turhan 210717044, Ali Seyhan 210717048

**Project report:** ai_final.pdf

## Project Overview  

This repository contains a chest X-ray disease detection system developed as part of an AI course final project by **Baki Turhan** and **Ali Seyhan**.  

The system utilizes two pre-trained deep learning models, **DenseNet121** and **VGG19**, to perform **multi-label classification** of chest X-ray images into 14 possible disease categories. The project demonstrates the practical use of transfer learning and deep convolutional neural networks in medical image analysis.

---

## Repository Structure  

```
final_ai/
│
├── ai_demo/ # Streamlit web application files
│ ├── app.py # Main application file
│ ├── Data_Entry_2017.csv # Metadata file for the Chest X-ray dataset
│
├── ai_final.ipynb # Jupyter notebook containing model training and evaluation
├── ai_final.pdf # Final project report (explains data preprocessing, model training, results)
├── proje_sunumu.pdf # Project presentation slides
```

---

## Disease Classes  

The models can detect the following chest diseases:

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
13. Pleural Thickening  
14. No Finding  

---

## Installation  

1. Install the required Python libraries:
   
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
