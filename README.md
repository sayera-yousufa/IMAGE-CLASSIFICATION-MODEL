# IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SAYERA YOUSUFA

*INTERN ID*: CT04DK237

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

#  Image Classification Using Convolutional Neural Networks (CNN)

##  Project Description

This project involves building a deep learning model for **image classification** using **Convolutional Neural Networks (CNNs)**. The model is trained on a labeled image dataset to classify images into predefined categories.

CNNs are especially powerful for analyzing visual imagery. They work by automatically learning spatial hierarchies of features through backpropagation, making them ideal for image-related tasks.

---

##  Dataset

The dataset used for this project consists of multiple classes of images. It is preprocessed and structured into training and testing directories.

- Images are resized to a standard dimension.
- Data augmentation is applied to improve generalization.
- Split: **Training** and **Validation** sets.

> You can modify the dataset path in the notebook if you're using a custom dataset.

---

##  Technologies Used

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV / PIL (for image preprocessing)

---

##  Model Architecture

The model is built using **Sequential API** of Keras with the following layers:

- `Conv2D` layers with ReLU activation
- `MaxPooling2D` for downsampling
- `Flatten` layer for converting 2D features to 1D
- `Dense` layers including final softmax layer for classification

> Optimizer: Adam  
> Loss Function: Categorical Crossentropy  
> Metrics: Accuracy

---

###  Prerequisites

Install the required libraries before running the notebook:
* tensorflow
* numpy
* matplotlib
* scikit-learn

## OUTPUT


##  RESULTS

-  **Training Accuracy**: 79.90%
-  **Test Accuracy**: 70.87%
-  **Test Loss**: 0.9173
The model shows a reasonable learning trend with increasing training accuracy and decreasing loss over epochs, indicating successful convergence.

##  Acknowledgements

This project was completed as part of my internship at **CodTech IT Solutions**.




