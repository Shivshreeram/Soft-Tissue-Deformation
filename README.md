# **Soft Tissue Deformation Analysis Using StrainNet**

## **Introduction to Soft Tissue Deformation**

Soft tissue deformation refers to the changes in the shape, structure, or volume of soft tissues (such as muscles, skin, organs) due to external forces or stress. This phenomenon is essential for various medical and biomechanical applications, including injury diagnosis, surgical planning, and prosthetics design. Accurately modeling soft tissue deformation is a challenging task due to the complex, nonlinear nature of soft tissues.

### **Why StrainNet?**

To model and predict soft tissue deformation, advanced methods like **StrainNet** offer a robust approach. StrainNet is a deep learning framework designed to analyze and predict soft tissue behavior under deformation. By leveraging synthetic datasets, StrainNet allows for accurate simulations and predictions in deformation mechanics, making it an ideal tool for this problem.

---

## **Step 1: Blob Detection - A Foundation for Soft Tissue Deformation Analysis**

![Blob-detection](../Blob-detection)

Our first step into soft tissue deformation analysis begins with a simplified task: **Blob Detection**. This task helps us understand the core concept of deformation by tracking the relative positions of particles or points (blobs). These blobs are used to represent the basic elements of soft tissue under deformation.

### **Why Blob Detection?**

Blob detection serves as a controlled and simplified environment where we can simulate and study basic deformation scenarios. By identifying and tracking the relative positions of particles, we can start mimicking the core principles of strain analysis, which will later help us tackle more complex soft tissue deformation problems.

---

## **Main Problem Statement: Soft Tissue Deformation Using StrainNet**

With a foundational understanding of basic deformation mechanics through blob detection, we now dive into the core of the problem: modeling **soft tissue deformation** more accurately. To achieve this, we utilize synthetic datasets that simulate the complex behavior of soft tissues under various deformation conditions.

The next step in our process is to download and utilize the **StrainNet** dataset from an external repository. This dataset provides realistic synthetic data that mimics the deformation of soft tissues, which is essential for training and evaluating advanced models like StrainNet.

You can find and download the StrainNet dataset from the following repository: [StrainNet Repository](https://github.com/reecehuff/StrainNet)

By working with these synthetic datasets, we can simulate a range of soft tissue deformation scenarios, allowing us to build, train, and refine a deep learning model that can predict and analyze the behavior of real-world soft tissue deformations.

---

#  Deformation Classification Using DeformationNet

To classify the type of deformation in soft tissues, I developed a model called **DeformationNet** using a Convolutional Neural Network (CNN), which is well-suited for image-based classification problems.

The model was trained on synthetic strain map datasets to classify each input into one of three deformation categories:

- **Compression**
- **Rigid**
- **Tension**

## 📊 Classification Results

- **Validation Accuracy:** 84.67%
- **Training Accuracy:** 98.2%
- **Overall Model Performance:** Consistent and stable learning with clear generalization.

### 🔹 Class-wise Metrics

| Class       | Precision | Recall | F1 Score |
|-------------|-----------|--------|----------|
| Compression | 90.2%     | 92.0%  | 91.1%    |
| Rigid       | 82.4%     | 84.0%  | 83.2%    |
| Tension     | 81.3%     | 78.0%  | 79.6%    |

The model shows highest performance on the **Compression** class. Most of the confusion occurs between **Rigid** and **Tension**, which can be addressed with further refinements.

---

## 🧩 Deformation Map Formation

After classifying the type of deformation, the next step is to generate **deformation maps** that represent the displacement across the soft tissue surface. This involves predicting pixel-wise displacements (strain fields) using U-Net.

### 🔬 U-Net-Based Deformation Prediction

To achieve this, I trained a **U-Net** model that takes in a differential grayscale image and outputs:

- **x-direction displacement map (`y_x`)**
- **y-direction displacement map (`y_y`)**

The model is designed as a deep U-Net with 6 encoder-decoder blocks and skip connections for spatial precision. Each output is supervised using Huber loss to balance smoothness and robustness.

### 📉 U-Net Training Results

- **Training Loss:** 0.28  
- **Validation Loss:** 0.36

This shows good generalization ability, and qualitative results indicate that the predicted displacement maps are spatially coherent and capture local deformation trends accurately.

---

By leveraging the **StrainNet** dataset, I am training a model that will predict and analyze soft tissue deformations and classify them as **compression**, **rigid**, or **tension**, while also generating the corresponding **displacement maps** using U-Net.
