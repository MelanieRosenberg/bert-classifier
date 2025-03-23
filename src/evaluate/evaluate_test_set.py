import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def evaluate_model(model_path, test_data_path, output_dir):
    """
    Evaluate the BERT model on test data
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data Excel file
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    test_df = pd.read_excel(test_data_path)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2  # Binary classification: list vs non-list
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Prepare data
    texts = test_df['text'].tolist()
    true_labels = test_df['is_list'].tolist()
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits).item()
            predictions.append(pred)
    
    # Calculate metrics
    report = classification_report(true_labels, predictions)
    print("\nClassification Report:")
    print(report)
    
    # Save report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-List', 'List'],
        yticklabels=['Non-List', 'List']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT list classifier')
    parser.add_argument('--model_path', required=True, help='Path to saved model')
    parser.add_argument('--test_data', required=True, help='Path to test data Excel file')
    parser.add_argument('--output_dir', default='output', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_data, args.output_dir)

if __name__ == '__main__':
    main() 