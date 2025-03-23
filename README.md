# BERT List Classifier

A model for classifying list items in documents using BERT fine-tuning.

## Project Structure

```
bert-list-classifier/
├── config/            # Configuration files
├── data/              # Data files (not added to github)
├── src/               # Source code
│   ├── data/          # Data processing scripts
│   ├── training/      # Training scripts
│   ├── evaluate/      # Model evaluation
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd bert-list-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Processing

The data processing pipeline consists of several steps:

1. Training Data Contamination:
```bash
python src/data/contaminate_data.py
```

2. Data Splitting and Balancing:
```bash
cd src/data/data_processor.py
```

## Training Process

The training process involves the following steps:

1. Data Loading: The system loads training data from an Excel file where each column represents a different category.
2. Data Preparation: The data is preprocessed and split into training and validation sets.
3. Model Training: The model is trained using a DeBERTa-v3-small architecture with the following features:
   - Mixed precision training with gradient accumulation
   - Learning rate scheduling with warmup
   - Early stopping based on validation loss
   - TensorBoard logging for monitoring
   - Checkpoint saving for best models
   - Graceful interruption handling

4. Model Evaluation: During training, the model is evaluated on the validation set with metrics including:
   - Accuracy
   - F1 Score (weighted)
   - Precision (weighted)
   - Recall (weighted)
   - Per-class performance metrics

## Training Configuration

The default training configuration includes:

- 10 epochs (with early stopping patience of 3)
- Batch size of 16
- Gradient accumulation steps of 4
- Learning rate of 2e-5
- 10% warmup ratio
- Weight decay of 0.01

## Running the Training

To train the model:
```bash
python src/train.py
```
This will:
1. Load data from data/train_data_contaminated.xlsx
2. Prepare and split the data
3. Train the model
4. Save model checkpoints and metrics in the output directory

## Monitoring Training

You can monitor the training progress with TensorBoard:
```bash
tensorboard --logdir=output/logs
```

## Model Evaluation

To evaluate the model on a test set:
```bash
python src/evaluate_test_set.py --model_path output/best_model.pt --test_data data/test_data.xlsx --output_dir evaluation_results
```
This will:
1. Load the trained model
2. Evaluate on the test data
3. Generate a classification report and confusion matrix
4. Save results to the specified output directory