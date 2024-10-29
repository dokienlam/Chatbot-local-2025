import torch

# Check if CUDA is available
print(torch.cuda.is_available())  # Should return True


print(torch.version.cuda)


# device = "cuda" if torch.cuda.is_available() else "cpu"
# inputs = input_ids["input_ids"].to(device)
