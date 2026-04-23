import os
import torch
import torch.optim as optim

# -----------------------------
# VALIDATION FUNCTION
# -----------------------------
def validate_model(model, val_loader, device):
    model.train()  # Faster R-CNN needs train mode for loss
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    return val_loss_sum / val_count if val_count > 0 else 0


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_model(model, train_loader, val_loader, device, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    best_val_loss = float('inf')

    # ✅ ensure folder exists
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{args.num_epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ✅ SAVE BEST MODEL
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print("✅ Best model saved!")

    # ✅ ALWAYS SAVE LAST MODEL (IMPORTANT)
    torch.save(model.state_dict(), 'checkpoints/last_model.pth')
    print("✅ Last model saved!")