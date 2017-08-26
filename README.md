# USPS_Digit_Classification

## Problem Description
This project introduces us to a major topic in Supervised Machine Learning, which is Classification. We shall be studying the multiclass classification problem, by applying supervised classification algorithms on two datasets provided:

  1. MNIST Dataset

  2. USPS Dataset
    
While solving the problem of Handwriting Recognition, we shall encounter the **NO FREE LUNCH THEOREM**. Our objective is to verify that this theorem holds even in this case, as we shall see in this implementation.

## Dataset

1. We have been provided with the MNIST handwritten digits dataset, which consists of 70000 grayscale images, representing 10 digits - 0 to 9. The images are each of 28 x 28 pixel resolution.

2. We shall be using the MNIST dataset for training our machine learning models.

3. The other dataset is the **USPS Dataset**. It consists of approximately 20000 pixels, which are at resolution of 100ppi. We need to preprocess these images to have same resolution as our MNIST data images, so that our trained models can be applied on this dataset.

## Implementation

In particular, we shall be applying the following algorithms for the task of Handwritten Character Recognition:
    
    1. Logistic Regression

    2. Single layered Neural Networks

    3. Convolutional Neural Networks
    
The objective is to prove the 'No Free Lunch' Theorem by training models on MNIST dataset, and testing on USPS dataset.

## No Free Lunch (NFL) Theorem

A model is a simplified representation of reality, and the simplifications are made to discard unnecessary detail and allow us to focus on the aspect of reality that we want to understand. These simplifications are grounded on assumptions; these assumptions may hold in some situations, but may not hold in other situations. This implies that a model that explains a certain situation well may fail in another situation.

The “No Free Lunch” theorem states that there is no one model that works best for every problem.

The assumptions of a great model for one problem may not hold for another problem, so it is common in machine learning to try multiple models and find one that works best for a particular problem.

This is especially true in supervised learning problem, which we are trying to solve in this project.

Validation or Cross-Validation is commonly used to assess the predictive accuracies of multiple models of varying complexity to find the best model.

**In our case, as results shall prove, the trained models performs exceptionally well on MNIST dataset, on which they were trained, but do not perform as well on USPS dataset, which the model has never seen before.**

## Results

1. Logistic Regression
 
    Accuracy on MNIST test -  92.25 %
    
    Accuracy on USPS data  -  36.30 %
    
2. Single Layer Neural Network

    Accuracy on MNIST test -  97.09 %
    
    Accuracy on USPS data  -  49.23 %
    
3. Convolutional Neural Networks

    Accuracy on MNIST test -  98.7 %
    
    Accuracy on USPS data  -  63.0 %
    
## Conclusion

We verified the 'No Free Lunch Theorem' in Machine Learning by testing the performance of our model on the USPS Dataset, and the model accuracy was way below what we achieved on the MNIST dataset.

Convolutional Neural networks seem the way forward, and we have verified the impact of Deep Learning on simple task of handwriting recognition. CNNs are performing some state of the art work on various Computer Vision tasks as well, in various fields such as medicine, automated cars, object detection, etc.

## Credits
This project was a submission for the course 'Intoduction to Machine Learning' taught by Prof. Sargur Srihari at State University of New York. I would like to thank him for teaching the course and providing the USPS dataset.
