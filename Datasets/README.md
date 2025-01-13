# Datasets Directory

This directory contains subdirectories for storing various datasets used in the project.

## Included Datasets:

### 1. FashionMNIST Dataset

- **Description:** FashionMNIST is a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
- **Download:** You can download the FashionMNIST dataset from [here](https://github.com/zalandoresearch/fashion-mnist#get-the-data).
- **Usage:** After downloading, extract the dataset and place it in the 'FashionMNIST' subdirectory.

### 2. CIFAR10 Dataset

- **Description:** CIFAR10 is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
- **Download:** You can download the CIFAR10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).
- **Usage:** After downloading, extract the dataset and place it in the 'CIFAR10' subdirectory.

### 3. sugarcane_damage_usa Dataset

- **Description:** sugarcane_damage_usa is a dataset of sugarcane damage images collected in the USA. It consists of images categorized into multiple damage types related to sugarcane.
- **Download:** Please refer to the source from where you obtained the sugarcane_damage_usa dataset for download instructions.
- **Usage:** After downloading, extract the dataset and place it in the 'sugarcane_damage_usa' subdirectory.

## Adding Custom Datasets:

If you have a custom dataset that does not have an inbuilt index like the ones provided, follow these steps to add it:

1. Create a Python class definition for your dataset, similar to the ones provided in the DataModules directory.
2. Save the dataset class definition file in the respective dataset directory.
3. Modify the DataModules.py file to include your custom dataset class and its corresponding DataModule.

For any issues or questions regarding the datasets, please refer to the respective sources or documentation.
