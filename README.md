# Image-to-Asymptote: Data Generation and Baseline Model Training

This project focuses on generating a dataset of mathematical figures (images) and their corresponding Asymptote code, and then training a ResNet-based image classification model to identify the type of mathematical function depicted. This serves as a foundational step for a larger goal of fine-tuning a language model for image-to-Asymptote code generation.

## Project Overview

The project involves several stages, managed through distinct Jupyter/Colab notebooks:

1.  **Data Generation & Preparation (`asymptote_data_gen.ipynb`):** Generates pairs of images and Asymptote code for various mathematical function types (linear, quadratic, circle, ellipse, hyperbola, sine, absolute value, tangent). It saves these to Google Drive, maintains a master metadata file, and splits this metadata into training and validation sets. This script is designed to be additive – it can append new samples to the existing dataset.
2.  **Saving Initial Model Weights (`load_model.ipynb`):** Creates an instance of an untrained ResNet50 model (approx. 25 million parameters with random initialization) and saves its initial state dictionary. This ensures a consistent starting point for both baseline testing and subsequent training.
3.  **Baseline Performance Testing (`baseline_testing.ipynb`):** Loads the saved *untrained* ResNet50 weights and evaluates its performance on the validation set. This establishes a baseline accuracy for classifying the function types from images before any training.
4.  **Model Training (`train.ipynb`):** Loads the same *untrained* ResNet50 weights, trains the model on the generated training dataset, validates it on the validation set, and saves the weights of the trained model (both the final epoch and the best performing one on validation).
5.  **Performance Comparison (Future - `test_performance.ipynb`):** (This notebook is planned) This notebook would be used to load the saved baseline (untrained) and trained model weights to perform a comparative analysis on a test set or the validation set.

## Prerequisites

*   **Google Colab:** The notebooks are designed to be run in a Google Colab environment.
*   **Google Drive:** Project files, datasets, and models are stored in Google Drive. Ensure your Drive is mounted correctly.
*   **Python Libraries:**
    *   `torch`, `torchvision`
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `matplotlib`
    *   `seaborn`
    *   `tqdm`
    These are generally pre-installed in Colab or can be installed via pip:
    ```bash
    !pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn tqdm -q
    ```
*   **Asymptote:** Required for data generation. The `asymptote_data_gen.ipynb` notebook includes a cell (Cell 0) to install it within the Colab environment. This needs to be run once per session if Asymptote is not already available.

## Directory Structure

All project files are expected to be within a main folder in your Google Drive, typically `My Drive/StarSparkProj/`.

```StarSparkProj/
├── AsymptoteDataset_Full/      # Generated images (.png) and Asymptote code (.asy)
│   ├── linear_s2.8_i0.25.png
│   ├── linear_s2.8_i0.25.asy
│   └── ... (many more files for different function types)
│
├── asymptote_data_gen.ipynb    # Notebook for data generation, augmentation, and splitting
├── load_model.ipynb            # Notebook to save initial untrained model weights
├── baseline_testing.ipynb      # Notebook for testing the untrained model
├── train.ipynb                 # Notebook for training the model
├── test_performance.ipynb      # (Planned) Notebook for final performance comparison
│
├── all_samples_metadata_Full.csv # Master metadata for all generated samples
├── train_metadata_Full.csv     # Training set metadata
├── val_metadata_Full.csv       # Validation set metadata
│
├── untrained_resnet50_initial_weights.pth # Saved initial weights of the ResNet50
├── trained_resnet50_model_v1_10epochs.pth # Final trained model weights (example name)
├── best_trained_resnet50_model_v1_10epochs.pth# Model weights with best validation accuracy
│
├── training_history_10epochs.csv    # CSV log of training/validation loss and accuracy per epoch
└── training_results_plot_10epochs.png # Plot of training/validation metrics


Usage Workflow
Follow these steps in order:
1. Initial Setup (Run once per Colab session if Asymptote isn't setup)
Open asymptote_data_gen.ipynb.
Run Cell 0 to install Asymptote and other dependencies like dvipng if not already present.
2. Data Generation and Preparation
Open asymptote_data_gen.ipynb.
Configure TOTAL_SAMPLES_GOAL (Cell 6):
The script appends to existing data. If you have 5000 samples and set the goal to 10000, it will try to generate 5000 new unique samples, distributed across all defined function types (including linear, quadratic, circle, ellipse, hyperbola, sine, and the newer absolute value, tangent).
Run all cells (Cell 0 through Cell 7).
Cell 0: Installs Asymptote.
Cell 1-4: Define imports, Asymptote templates, parameter generation functions, and compilation logic.
Cell 5: Defines the FUNCTION_GENERATORS (including new shapes) and the batch generation logic.
Cell 6: Executes generate_large_batch. This populates AsymptoteDataset_Full/ with .png and .asy files and updates/creates all_samples_metadata_Full.csv. This can be time-consuming.
Cell 7: Splits the (potentially updated) all_samples_metadata_Full.csv into train_metadata_Full.csv and val_metadata_Full.csv.
Outputs:
Images and .asy files in StarSparkProj/AsymptoteDataset_Full/.
StarSparkProj/all_samples_metadata_Full.csv (updated).
StarSparkProj/train_metadata_Full.csv (recreated).
StarSparkProj/val_metadata_Full.csv (recreated).
3. Save Initial Untrained Model Weights
Open load_model.ipynb.
Run all cells (Cell 1 to Cell 3).
Cell 1: Mounts Drive, imports.
Cell 2: Sets path for saving initial weights.
Cell 3: Instantiates a ResNet50 model with random weights and saves its state_dict.
Output:
StarSparkProj/untrained_resnet50_initial_weights.pth
4. Baseline Performance Testing (Untrained Model)
Open baseline_testing.ipynb.
Ensure paths in Cell 2 (BASE_DIR, METADATA_FILE, UNTRAINED_MODEL_WEIGHTS_PATH) are correct.
Run all cells (Cell 1 to Cell 7).
Cell 4 loads val_metadata_Full.csv.
Cell 5 loads the ResNet50 model structure and then loads the untrained weights from untrained_resnet50_initial_weights.pth. It then adapts the final classification layer to your dataset's number of classes.
Cell 7 evaluates the model on the validation set.
Outputs:
Console output showing accuracy, classification report, and confusion matrix for the untrained model. (Note: Missing image files listed in the metadata will be skipped, and the report will reflect evaluation only on successfully loaded images).
5. Model Training
Open train.ipynb.
Configure training parameters in Cell 2:
NUM_EPOCHS (e.g., 10 or as desired).
Verify INITIAL_MODEL_WEIGHTS_PATH points to your saved untrained weights.
TRAINED_MODEL_SAVE_PATH and BEST_MODEL_SAVE_PATH are set (the script will add epoch info to the filenames).
Run all cells (Cell 1 to Cell 6).
Cell 4: Loads train_metadata_Full.csv and val_metadata_Full.csv, prepares DataLoaders.
Cell 5: Defines the model, loads the initial untrained weights from INITIAL_MODEL_WEIGHTS_PATH, defines optimizer and loss.
Cell 6: Executes the training loop. This will train the model, validate it after each epoch, and save the best performing model based on validation accuracy.
Outputs:
StarSparkProj/trained_resnet50_model_v1_...epochs.pth (model weights from the final epoch).
StarSparkProj/best_trained_resnet50_model_v1_...epochs.pth (model weights with the highest validation accuracy).
StarSparkProj/training_history_...epochs.csv (CSV log of metrics).
StarSparkProj/training_results_plot_...epochs.png (Plot of training/validation accuracy and loss).
6. (Future) Performance Comparison
Open a new notebook (e.g., test_performance.ipynb).
Load the untrained model weights (from untrained_resnet50_initial_weights.pth) and evaluate it on the validation set (similar to baseline_testing.ipynb).
Load the trained model weights (e.g., from best_trained_resnet50_model_v1_...epochs.pth) and evaluate it on the same validation set.
Compare the performance metrics (accuracy, precision, recall, F1-score) between the untrained and trained models.
Key Generated Files Summary
Dataset: StarSparkProj/AsymptoteDataset_Full/ (images & .asy files)
Metadata:
StarSparkProj/all_samples_metadata_Full.csv
StarSparkProj/train_metadata_Full.csv
StarSparkProj/val_metadata_Full.csv
Model Weights:
StarSparkProj/untrained_resnet50_initial_weights.pth
StarSparkProj/best_trained_resnet50_model_v1_...epochs.pth
StarSparkProj/trained_resnet50_model_v1_...epochs.pth
Training Logs:
StarSparkProj/training_history_...epochs.csv
StarSparkProj/training_results_plot_...epochs.png
Notes and Considerations
Google Drive I/O: Reading many small image files from Google Drive can be slow during training and evaluation. Ensure a stable internet connection. Setting num_workers=0 in DataLoader can make console output cleaner for debugging but slows down data loading; for actual training, num_workers=2 or more is usually better.
Missing Image Files: The scripts attempt to handle missing image files by skipping them during data loading and evaluation. The final evaluation reports will indicate how many samples were successfully processed. It's crucial to ensure your metadata CSVs accurately reflect the available image files for reliable metrics.
Model Scope: The current ResNet50 model is trained as an image classifier to predict the type of mathematical function (linear, quadratic, etc.) from an image. This is a step towards the larger project goal of generating Asymptote code.
Additive Data Generation: Running asymptote_data_gen.ipynb multiple times with the same DATA_DIR and METADATA_FILENAME will append new unique samples (up to TOTAL_SAMPLES_GOAL) rather than overwriting, allowing for dataset expansion over time.
