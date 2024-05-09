import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModel

# Load BioMedLM tokenizer
biomed_tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")

# Load BioMedLM model
biomed_model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

# Load LinkBERT tokenizer
linkbert_tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")

# Load LinkBERT model
linkbert_model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")

class BiomedicalTextClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(BiomedicalTextClassifier, self).__init__()
        
        # BioMedLM model
        self.biomed_model = biomed_model
        
        # LinkBERT model
        self.linkbert_model = linkbert_model
        
        # Output layer for classification
        self.output_layer = torch.nn.Linear(self.biomed_model.config.hidden_size + self.linkbert_model.config.hidden_size, num_classes)
        
    def forward(self, biomed_inputs, linkbert_inputs):
        # Forward pass through BioMedLM
        biomed_outputs = self.biomed_model(**biomed_inputs)[0][:, 0, :]  # Take only the CLS token
        
        # Forward pass through LinkBERT
        linkbert_outputs = self.linkbert_model(**linkbert_inputs)[0][:, 0, :]  # Take only the CLS token
        
        # Concatenate the outputs of BioMedLM and LinkBERT
        combined_outputs = torch.cat((biomed_outputs, linkbert_outputs), dim=1)
        
        # Feed combined outputs to the output layer for classification
        logits = self.output_layer(combined_outputs)
        
        return logits

num_classes = 2  # Number of classes in your downstream task
model = BiomedicalTextClassifier(num_classes)

# Assuming you have your inputs prepared
biomed_inputs = {"input_ids": ..., "attention_mask": ...}  # BioMedLM input
linkbert_inputs = {"input_ids": ..., "attention_mask": ...}  # LinkBERT input

# Forward pass
logits = model(biomed_inputs, linkbert_inputs)

import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

class BiomedicalTextClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout_prob=0.1):
        super(BiomedicalTextClassifier, self).__init__()
        
        self.biomed_model = biomed_model
        self.linkbert_model = linkbert_model
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.output_layer = torch.nn.Linear(
            self.biomed_model.config.hidden_size + self.linkbert_model.config.hidden_size,
            num_classes
        )
        
    def forward(self, biomed_inputs, linkbert_inputs):
        biomed_outputs = self.biomed_model(**biomed_inputs)[0][:, 0, :]
        linkbert_outputs = self.linkbert_model(**linkbert_inputs)[0][:, 0, :]
        combined_outputs = torch.cat((biomed_outputs, linkbert_outputs), dim=1)
        combined_outputs = self.dropout(combined_outputs)
        logits = self.output_layer(combined_outputs)
        return logits

# Initialize model
num_classes = 2
model = BiomedicalTextClassifier(num_classes)

# Define optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        biomed_inputs, linkbert_inputs, labels = batch
        logits = model(biomed_inputs, linkbert_inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            biomed_inputs, linkbert_inputs, labels = batch
            logits = model(biomed_inputs, linkbert_inputs)
            loss = F.cross_entropy(logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_dataloader)
    accuracy = correct / total
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {val_loss}, Accuracy: {accuracy}")



