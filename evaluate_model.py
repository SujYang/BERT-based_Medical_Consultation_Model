from sklearn.metrics import accuracy_score

# Evaluation loop
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        biomed_inputs, linkbert_inputs, labels = batch
        logits = model(biomed_inputs, linkbert_inputs)
        _, predicted = torch.max(logits, 1)
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy}")
