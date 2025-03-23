import os
import json
import random
import numpy as np
import pandas as pd
import torch
import signal
import sys
import shutil
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set the environment variable to control tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 10,
    'batch_size': 16,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'max_length': 128,
    'early_stopping_patience': 3
}

class ListDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=TRAINING_CONFIG['max_length']):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, is_best, filename='output/checkpoint.pt'):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'training_config': TRAINING_CONFIG
    }, filename)
    if is_best:
        shutil.copyfile(filename, 'output/best_model.pt')

def train_model(model, train_loader, val_loader, device, writer, label_names, num_epochs=TRAINING_CONFIG['num_epochs']):
    """Train the model."""
    global_epoch = 0  # Make epoch accessible to signal handler
    global_val_loss = float('inf')  # Initialize val_loss for signal handler
    
    # Initialize optimizer and scaler for mixed precision training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scaler = GradScaler()
    
    # Calculate total training steps for scheduler
    num_training_steps = len(train_loader) * num_epochs // TRAINING_CONFIG['gradient_accumulation_steps']
    num_warmup_steps = int(num_training_steps * TRAINING_CONFIG['warmup_ratio'])
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_val_loss = float('inf')
    no_improvement = 0
    
    # Set up signal handler for graceful interruption
    def signal_handler(sig, frame):
        print('Training interrupted! Saving checkpoint...')
        save_checkpoint(model, optimizer, scheduler, global_epoch, global_val_loss, False, 'output/interrupted_checkpoint.pt')
        writer.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    for epoch in range(num_epochs):
        global_epoch = epoch  # Update global epoch for signal handler
        
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / TRAINING_CONFIG['gradient_accumulation_steps']
            
            # Print batch loss periodically
            if step % 10 == 0:
                print(f'  Batch {step}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            scaler.scale(loss).backward()
            
            if (step + 1) % TRAINING_CONFIG['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * TRAINING_CONFIG['gradient_accumulation_steps']
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Store the global val_loss for signal handler
        global_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        # Generate and print classification report
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=label_names,
            digits=3,
            zero_division=0
        )
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/validation', val_loss / len(val_loader), epoch)
        writer.add_scalar('Metrics/accuracy', accuracy, epoch)
        writer.add_scalar('Metrics/f1', f1, epoch)
        writer.add_scalar('Metrics/precision', precision, epoch)
        writer.add_scalar('Metrics/recall', recall, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        print(f'Epoch {epoch + 1}:')
        print(f'  Training Loss: {train_loss / len(train_loader):.4f}')
        print(f'  Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  F1 Score: {f1:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
        print("\nClassification Report:")
        print(report)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, True)
        else:
            no_improvement += 1
            if no_improvement >= TRAINING_CONFIG['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save regular checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, False)

def load_and_prepare_data(data_path):
    """Load and preprocess the data from Excel format."""
    print(f"Loading data from {data_path}")
    df = pd.read_excel(data_path)
    
    # Print sample of data for verification
    print("\nData sample:")
    print(df.head())
    
    # Get the column names - these are the label categories
    columns = df.columns.tolist()
    print(f"\nFound {len(columns)} categories: {columns}")
    
    # Prepare data for training
    all_texts = []
    all_labels = []
    
    # For each column, extract all values and add them with the column name as the label
    for col_idx, column in enumerate(columns):
        # Extract non-empty values from this column
        column_values = df[column].dropna().astype(str).tolist()
        print(f"Category '{column}' has {len(column_values)} examples")
        
        # Add each value as a text example with its column as the label
        all_texts.extend(column_values)
        all_labels.extend([col_idx] * len(column_values))
    
    # Create label mapping (column name to index)
    label_mapping = {column: idx for idx, column in enumerate(columns)}
    
    print(f"\nTotal examples: {len(all_texts)}")
    print(f"Label mapping: {label_mapping}")
    
    # Return all data
    return all_texts, all_labels, label_mapping

def main():
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/logs', exist_ok=True)
    
    # Load and prepare data
    texts, labels, label_mapping = load_and_prepare_data('data/train_data_contaminated.xlsx')
    
    # Save label mapping for future use
    with open('output/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Get label names for reporting
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda x: x[1])]
    
    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED
    )
    
    print(f"\nTraining set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    
    # Save split info for future reference
    split_info = {
        'train_size': len(train_texts),
        'val_size': len(val_texts),
        'seed': SEED
    }
    with open('output/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Save training configuration
    with open('output/training_config.json', 'w') as f:
        json.dump(TRAINING_CONFIG, f, indent=2)
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    num_labels = len(label_mapping)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    # Create datasets and dataloaders
    train_dataset = ListDataset(train_texts, train_labels, tokenizer)
    val_dataset = ListDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for easier debugging
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0  # Set to 0 for easier debugging
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='output/logs')
    
    # Train model
    print("\nTraining model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Train the model
    train_model(model, train_loader, val_loader, device, writer, label_names)
    
    # Close TensorBoard writer
    writer.close()
    
    print("\nTraining completed!")
    print("Outputs saved in 'output' directory:")
    print("- training_config.json: Contains the training configuration")
    print("- label_mapping.json: Contains the mapping of label strings to indices")
    print("- split_info.json: Contains the train/validation split information")
    print("- best_model.pt: The best model checkpoint with optimizer and scheduler states")
    print("- checkpoint.pt: The last model checkpoint")
    print("- logs/: Contains TensorBoard logs for visualization")

if __name__ == '__main__':
    main()