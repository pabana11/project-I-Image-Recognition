import torch

def validate_model(model, val_loader, device):
    # We keep the model in train mode to get the loss_dict
    # but we use torch.no_grad() so it doesn't learn from validation data
    model.train() 
    val_loss = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [
                {
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device)
                }
                for t in targets
            ]

            loss_dict = model(images, targets)
            
            # Now loss_dict will be a dictionary, not a list!
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item()

    return val_loss / len(val_loader)