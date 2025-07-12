# Traffic Sign Recognition: Exploring Self-Supervised Vision Transformers and CNNs

This project explores two distinct approaches for working with the German Traffic Sign Recognition Benchmark (GTSRB) dataset:

1.  **Self-Supervised Learning with a Vision Transformer (ViT)**: This section implements a DINO-like self-supervised learning framework to generate robust image embeddings without explicit labels. These embeddings can then be used for tasks like similarity search or as pre-trained features.
2.  **Supervised Traffic Sign Classification with a Convolutional Neural Network (CNN)**: This section builds a traditional CNN model for classifying traffic signs based on their respective categories using a supervised learning paradigm.

-----

## Part 1: Self-Supervised Vision Transformer for Image Embeddings

This part of the project focuses on learning rich image representations in a self-supervised manner using a Vision Transformer, inspired by the DINO (Self-Distillation with No Labels) framework. The goal is to obtain embeddings where similar images are close in the embedding space.

### Features

  * **Patch-based Image Processing**: Images are divided into fixed-size patches, which are then flattened into sequences for the Transformer.
  * **Student-Teacher Architecture**: A student network is trained to match the output of a teacher network, which is an exponentially moving average of the student.
  * **Contrastive Loss**: Uses a custom loss function (`HLoss`) to encourage similarity between augmented views of the same image and diversity across different images.
  * **WandB Integration**: Logs training metrics and visualizes embedding results using Weights & Biases.

### Installation

Ensure you have the necessary libraries installed for this part:

```bash
pip install deepspeed wandb pytorch_lightning torch torchvision matplotlib scikit-learn tqdm
```

### Data Preparation

The `ImageData` and `ImageOriginalData` classes handle loading images from the specified `TRAIN_FILES` path (which points to the GTSRB Test set in this context) and applying different augmentations or resizing. The `CollateFn` prepares these images by splitting them into patches and normalizing pixel values.

### Model Architecture

The core model (`Model`) is a custom Vision Transformer:

  * It takes flattened image patches as input.
  * Includes learnable **positional embeddings** and a **CLS token**.
  * Uses a `TransformerEncoder` for feature extraction.
  * Projects and normalizes the output.

The `HLoss` class implements a DINO-like loss, which computes the cross-entropy between the softmax output of the teacher (centered and sharpened) and the log-softmax output of the student.

### Training

The `LightningModel` orchestrates the training process using PyTorch Lightning:

  * Manages the **student and teacher networks**.
  * Implements the **loss calculation** and optimization step.
  * Updates the **center vector** (for `HLoss`) and the **teacher parameters** using exponential moving averages.
  * Logs training and validation results to Weights & Biases, including visualizations of the **closest image pairs** based on learned embeddings during validation epochs.

### Embedding Visualization

After training, the model can generate embeddings for images. A utility function `plot_closest_pairs` is used to visualize an input image and its `TOPK` most similar images from the dataset based on their embeddings, providing insight into the learned representations.

-----

## Part 2: Supervised Traffic Sign Classification with CNN

This part of the project focuses on building and training a Convolutional Neural Network for classifying traffic signs from the GTSRB dataset.

### Installation

For this part, you'll need:

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn Pillow
```

### Data Collection & Exploration

  * The project loads traffic sign images from the `Train` and `Test` directories of the GTSRB dataset.
  * It also utilizes `labels.csv` to map class IDs to traffic sign names.
  * Initial visualization includes showing random sample images from the test set and plotting the distribution of image dimensions.

### Data Preprocessing

  * All training images are resized to a uniform `(50, 50)` resolution.
  * Pixel values are normalized to a `[0, 1]` range by dividing by 255.
  * The processed images and their corresponding labels are saved as NumPy arrays (`Training_set.npy`, `Label_Id.npy`).
  * The training data is split into training and validation sets (80% train, 20% validation).
  * Target labels are converted to a **one-hot encoded** format for compatibility with the model's output layer.
  * The distribution of images across different classes is visualized to check for class imbalance.

### Model Building

A sequential Convolutional Neural Network (CNN) is constructed using `tensorflow.keras.models.Sequential`:

  * It consists of **three convolutional blocks**, each followed by a `MaxPool2D` layer and a `Dropout` layer to prevent overfitting.
  * A `Flatten` layer converts the 2D feature maps into a 1D vector.
  * Followed by a `Dense` hidden layer with `relu` activation and another `Dropout` layer.
  * The final `Dense` output layer has 43 units (one for each traffic sign class) with a `softmax` activation function for multi-class classification.
  * The model is compiled using `sparse_categorical_crossentropy` loss (as `y_train` is integer-encoded during `model.fit`, despite `y_train_cat` being generated), `adam` optimizer, and `accuracy` as the metric.

### Training

The CNN model is trained on the prepared training data:

  * **Epochs**: 25 epochs are set as the maximum.
  * **Batch Size**: 64 images per batch.
  * **Validation Data**: Used to monitor performance on unseen data during training.
  * **Early Stopping**: Training stops if the validation loss does not improve for 2 consecutive epochs, preventing overfitting and saving computation.

### Model Evaluation & Prediction

  * Training and validation accuracy and loss are plotted over epochs to observe model performance and identify potential overfitting.
  * The trained CNN model is loaded and used to make predictions on the separate test dataset.
  * A **classification report** is generated using `sklearn.metrics.classification_report` to provide detailed metrics (precision, recall, f1-score) for each class and overall accuracy.

-----

## Dataset

This project utilizes the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

  * **Source**: The dataset is expected to be in the `../input/gtsrb-german-traffic-sign` directory.
  * **Structure**: It contains `Train` and `Test` image directories, along with `labels.csv`, `Train.csv`, and `Test.csv` files providing metadata and ground truth.

-----

## Usage

To run this project:

1.  **Download the GTSRB dataset** and place it in the expected directory structure, typically `../input/gtsrb-german-traffic-sign/`.
2.  **Install all required libraries** as listed in the respective sections.
3.  **Execute the code sections sequentially**:
      * The first part (Vision Transformer) will train the self-supervised model and visualize embeddings. Note that it specifically uses the `Test` folder from the GTSRB dataset for training its embeddings.
      * The second part (CNN Classification) will preprocess the training data, train the CNN model, and evaluate its performance on the GTSRB test set.

This README provides a comprehensive guide to understanding and reproducing the results of this traffic sign recognition project.
