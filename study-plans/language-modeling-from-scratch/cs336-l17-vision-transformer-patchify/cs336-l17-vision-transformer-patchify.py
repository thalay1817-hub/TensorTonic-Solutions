import torch

def vision_transformer_patchify(images, patch_height, patch_width):
    dtype = images.dtype
    device = images.device

    batch, channels, height, width = images.shape

    grid_h = height // patch_height
    grid_w = width // patch_width

    # Reshape into grid and local-patch axes:
    # (B, C, H, W) -> (B, C, grid_h, patch_height, grid_w, patch_width)
    reshaped = images.reshape(batch, channels, grid_h, patch_height, grid_w, patch_width)

    # Permute to bring grid axes first (row-major over grid_h, grid_w),
    # then channel, local row, local column together for flattening:
    # -> (B, grid_h, grid_w, C, patch_height, patch_width)
    permuted = reshaped.permute(0, 2, 4, 1, 3, 5)

    # Flatten grid dims into patch_count, and (C, patch_height, patch_width) into feature dim
    tokens = permuted.reshape(batch, grid_h * grid_w, channels * patch_height * patch_width)
    tokens = tokens.to(dtype=dtype, device=device)

    # Build coordinates in row-major grid order using meshgrid
    row_idx, col_idx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.int64, device=device),
        torch.arange(grid_w, dtype=torch.int64, device=device),
        indexing="ij",
    )
    coordinates = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)], dim=1)
    coordinates = coordinates.to(dtype=torch.int64, device=device)

    return {
        "tokens": tokens,
        "coordinates": coordinates,
    }