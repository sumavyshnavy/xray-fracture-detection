# X-Ray Fracture Detection using Machine Learning

## Overview
This project focuses on detecting bone fractures in X-ray images using a machine learning-based approach. The objective is to classify X-ray images as fractured or non-fractured by extracting relevant features and training a classification model.

---

## Tech Stack
- Python
- OpenCV
- NumPy
- Scikit-learn

---

## Methodology
1. Data Preprocessing  
   - Converted X-ray images to grayscale  
   - Applied resizing and normalization  
   - Reduced noise to improve feature clarity  

2. Feature Extraction  
   - Extracted meaningful patterns using image processing techniques  
   - Focused on structural differences between fractured and normal bones  

3. Model Training  
   - Trained a Support Vector Machine (SVM) classifier  
   - Tuned parameters for improved performance  

4. Evaluation  
   - Evaluated on test data  
   - Achieved an accuracy of approximately 91%  

---

## Results
- Successfully classifies fractured vs non-fractured X-rays  
- Achieved ~99% accuracy  
- Demonstrates potential for assisting in early-stage fracture detection  

---

## Sample Output
<img width="1080" height="3535" alt="xray2" src="https://github.com/user-attachments/assets/67dd5699-00e4-4e26-b50b-0a38417ca988" />
<img width="1080" height="3690" alt="xray1" src="https://github.com/user-attachments/assets/b62736ac-8570-45a8-8760-452c0edb16cf" />


---

## Key Learnings
- Understanding of medical image processing  
- Experience with feature extraction and classification  
- Practical application of SVM and evaluation metrics  

---

## Future Improvements
- Implement deep learning models (CNNs)  
- Increase dataset size for better generalization  
- Add real-time detection capability  
- Build a simple user interface  

---

## Note
This project was developed for learning and experimentation in computer vision and machine learning.
