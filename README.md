Brain Tumor Segmentation using Residual U-Net
Table of Contents
Introduction

Features

Dataset

Installation

Usage

Model Architecture

Loss Functions and Metrics

Data Augmentation

Results

Contributing

License

Citations

Contact

Introduction
This project implements a deep learning solution for brain tumor segmentation from multi-modal MRI scans using a Residual U-Net architecture. The goal is to accurately segment different tumor sub-regions (necrotic, edema, and enhancing tumor) from T1, T1ce, T2, and FLAIR MRI sequences. This model is built using PyTorch and leverages various data preprocessing and augmentation techniques to improve performance.

Features
Multi-modal MRI Support: Processes T1, T1ce, T2, and FLAIR MRI sequences.

Residual U-Net Architecture: Utilizes a U-Net with residual blocks for enhanced feature extraction and improved gradient flow.

Data Augmentation: Includes rotation, flipping, brightness, and contrast adjustments to increase data variability and model robustness.

Combined Loss Function: Employs a combination of Dice Loss and Binary Cross-Entropy (BCE) Loss for effective segmentation training.

Kaggle Integration: Includes steps to easily download the BraTS 2020 dataset directly from Kaggle.

Visualization: Provides utilities to visualize input MRI slices, ground truth masks, and model predictions.

Dataset
The project uses the BraTS 2020 (Brain Tumor Segmentation) dataset. This dataset consists of multi-modal MRI scans of glioblastoma (GBM) and lower-grade glioma (LGG) patients, along with expert-annotated tumor segmentations.

The dataset includes the following modalities for each patient:

T1: Native T1-weighted

T1ce: Post-contrast T1-weighted

T2: T2-weighted

FLAIR: Fluid Attenuated Inversion Recovery

SEG: Ground truth segmentation mask (labels: 0 for background, 1 for necrotic and non-enhancing tumor core, 2 for peritumoral edema, and 4 for GD-enhancing tumor). For this project, these are converted to a binary mask (0: background, 1: any tumor).

The notebook includes a script to download and extract the dataset from Kaggle.

Installation
To set up the project locally, follow these steps:

Clone the repository (if applicable):

git clone <your-repository-url>
cd <your-project-directory>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate # On Windows: `venv\Scripts\activate`

Install the required packages:
The necessary libraries are listed in the provided Jupyter notebook. You can install them using pip:

pip install numpy pandas matplotlib opencv-python nibabel scikit-learn torch torchvision
pip install kaggle # Only if you need to download the dataset via Kaggle API

Note: Ensure you have a compatible CUDA installation if you plan to use a GPU for training.

Download the dataset:
The Jupyter notebook (segmentation_res_unet.ipynb) contains a section to download the BraTS 2020 dataset from Kaggle. You will need a Kaggle API token (kaggle.json) to do this. Follow the instructions in the notebook under "Download data from Kaggle".

Usage
Open the Jupyter Notebook:

jupyter notebook segmentation_res_unet.ipynb

Run the cells sequentially:
The notebook is structured to guide you through the entire process:

Setup: Installs dependencies and downloads the dataset.

Configuration: Defines hyperparameters and paths.

Data Preprocessing: Functions for loading, normalizing, and extracting 2D slices.

Dataset and DataLoader: Sets up custom dataset and PyTorch DataLoaders.

Model Definition: Defines the Residual Double Convolution Block and the ResUNet model.

Loss and Metrics: Defines Dice Loss and Combined Loss, along with Dice Coefficient metric.

Training and Validation Functions: Implements the training and validation loops.

Training Execution: Runs the training process, saves the model, and plots training curves.

Prediction and Visualization: Shows how to make predictions and visualize results.

Adjust Configuration:
You can modify the Config class parameters (e.g., IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE) at the beginning of the notebook to experiment with different settings.

Model Architecture
The core of the segmentation model is a Residual U-Net.

ResidualDoubleConv Block
This custom block forms the building unit of the U-Net. It consists of:

Two convolutional layers with a kernel size of 3x3 and padding of 1.

Batch Normalization after each convolution.

ReLU activation function.

Dropout for regularization.

A residual (shortcut) connection that adds the input x directly to the output of the second convolutional layer. This helps in training deeper networks by mitigating the vanishing gradient problem.

ResUNet
The ResUNet follows the traditional U-Net encoder-decoder structure with the ResidualDoubleConv blocks replacing the standard convolutional blocks.

Encoder (Downsampling Path):

Consists of multiple ResidualDoubleConv blocks followed by MaxPool2d for downsampling.

Each block increases the number of features (e.g., 64, 128, 256, 512).

Skip connections are established from the output of each encoder block to the corresponding decoder block.

Bottleneck:

A ResidualDoubleConv block at the deepest point of the network.

Decoder (Upsampling Path):

Uses ConvTranspose2d for upsampling, followed by ResidualDoubleConv blocks.

The skip connections from the encoder are concatenated with the upsampled features, allowing the decoder to recover fine-grained details lost during downsampling.

Output Layer:

A final 1x1 convolution maps the features to a single channel.

A sigmoid activation function is applied to produce a probability map for binary segmentation.

Loss Functions and Metrics
Combined Loss
The model is trained using a CombinedLoss function, which is a weighted sum of:

Dice Loss: This loss function is particularly effective for image segmentation tasks, especially when dealing with imbalanced classes (e.g., small tumor regions). It measures the similarity between the predicted segmentation and the ground truth.


L 
Dice
​
 =1− 
∣X∣+∣Y∣
2⋅∣X∩Y∣
​
 

where X is the predicted mask and Y is the ground truth mask.

Binary Cross-Entropy (BCE) Loss: A standard loss function for binary classification problems, which penalizes pixel-wise prediction errors.


L 
BCE
​
 =−[ylog( 
y
^
​
 )+(1−y)log(1− 
y
^
​
 )]

where y is the true label and 
haty is the predicted probability.

The CombinedLoss is defined as:


L 
Combined
​
 =α⋅L 
Dice
​
 +(1−α)⋅L 
BCE
​
 

where 
alpha is a weighting factor (defaulted to 0.5 in the configuration).

Dice Coefficient
The Dice Coefficient (or F1-score) is used as the primary evaluation metric. It quantifies the similarity between the predicted segmentation and the ground truth, ranging from 0 (no overlap) to 1 (perfect overlap).

Data Augmentation
To enhance the model's generalization capabilities and reduce overfitting, the following data augmentation techniques are applied during training with a specified probability:

Random Rotation: Rotates both the image and mask by a random angle within a defined range (e.g., 
pm15 degrees).

Random Horizontal Flip: Flips the image and mask horizontally.

Random Vertical Flip: Flips the image and mask vertically.

Random Brightness and Contrast Adjustment: Randomly adjusts the brightness and contrast of the MRI modalities.

These augmentations are applied only to the training dataset.

Results
After training, the model's performance can be evaluated using the Dice Coefficient. The notebook also provides functionality to visualize predictions against the ground truth masks for qualitative assessment. Training and validation loss/Dice curves are plotted to monitor the training progress.

Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

License
This project is open-source and available under the MIT License.

Citations
Please cite the following if you use this dataset or approach in your work:

BraTS 2020 Dataset:

(You will need to find the specific citation for BraTS 2020. A common approach is to look for the "About" or "Data" section on the official BraTS challenge website or the Kaggle dataset page for citation instructions. It usually involves citing a primary research paper.)

Example (placeholder - please replace with actual citation):

Menze, B. H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE Transactions on Medical Imaging 34.10 (2015): 1993-2024.

(Add any other relevant BraTS 2020 specific citations here.)

U-Net Architecture:

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention—MICCAI 2015 (pp. 234-241). Springer, Cham.

Residual Connections (if directly referencing a specific paper):

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

Contact
For any questions or inquiries, please contact [Your Name/Email/GitHub Profile].
