from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

def train_and_validate(model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    print(f'Training with: {model_name}')
    train_losses = []
    val_losses = []
    epoch_val_losses = []
    epoch_true_labels = []
    epoch_predicted_labels = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        # Average training loss for the current epoch
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_losses.clear()
        epoch_true_labels.clear()
        epoch_predicted_labels.clear()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                epoch_val_losses.append(loss.item())

                preds = (torch.sigmoid(outputs) > 0.5).float()
                epoch_true_labels.extend(labels.cpu().numpy())
                epoch_predicted_labels.extend(preds.cpu().numpy())

        # Average validation loss for the current epoch
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)

        accuracy = accuracy_score(epoch_true_labels, epoch_predicted_labels)
        f1 = f1_score(epoch_true_labels, epoch_predicted_labels)
        precision = precision_score(epoch_true_labels, epoch_predicted_labels)
        recall = recall_score(epoch_true_labels, epoch_predicted_labels)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return train_losses, val_losses, epoch_true_labels, epoch_predicted_labels
