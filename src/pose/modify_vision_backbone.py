import torch

# Path to the original checkpoint
original_checkpoint_path = 'best_EPE_epoch_182.pth'

# Path to save the filtered backbone checkpoint
backbone_checkpoint_path = 'best_EPE_epoch_182_backbone.pth'

# Load the original checkpoint
checkpoint = torch.load(original_checkpoint_path, map_location='cpu')

# Inspect keys in the checkpoint (use this to identify backbone keys if unsure)
print("Checkpoint keys:", checkpoint.keys())

# Extract the state_dict
state_dict = checkpoint['state_dict']  # Adjust key as necessary

# Filter keys for the backbone only
backbone_state_dict = {
    k.replace('backbone.', ''): v  # Remove prefix if needed
    for k, v in state_dict.items() if 'backbone.' in k
}

# Save the filtered backbone state_dict as a new checkpoint
torch.save({'state_dict': backbone_state_dict}, backbone_checkpoint_path)

print(f"Filtered backbone checkpoint saved to {backbone_checkpoint_path}")
