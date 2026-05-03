import torch


def compute_stream_pearson(X, window_length, layout="stream_major"):
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    X = X.contiguous()
    num_nodes, num_features = X.shape

    if num_nodes % window_length != 0:
        raise ValueError("num_nodes must be divisible by window_length")

    num_streams = num_nodes // window_length

    if layout == "stream_major":
        X_stream = X.view(num_streams, window_length, num_features)
    elif layout == "time_major":
        X_stream = X.view(window_length, num_streams, num_features).transpose(0, 1)
    else:
        raise ValueError("layout must be 'stream_major' or 'time_major'")

    X_stream_flat = X_stream.reshape(num_streams, -1)
    mean = X_stream_flat.mean(dim=1, keepdim=True)
    std = X_stream_flat.std(dim=1, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    X_std = (X_stream_flat - mean) / std

    pearson_matrix = (X_std @ X_std.t()) / X_std.shape[1]
    return pearson_matrix, num_streams


def build_similarity_graph(
    X,
    pearson_matrix,
    window_length,
    k,
    delta,
    layout="stream_major",
):
    X = X.contiguous()
    num_nodes, _ = X.shape

    if num_nodes <= 1 or k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=X.device)

    if num_nodes % window_length != 0:
        raise ValueError("num_nodes must be divisible by window_length")

    num_streams = num_nodes // window_length
    if pearson_matrix.shape[0] != num_streams:
        raise ValueError("pearson_matrix size does not match stream count")

    if layout == "stream_major":
        stream_id = torch.arange(num_nodes, device=X.device) // window_length
    elif layout == "time_major":
        stream_id = torch.arange(num_nodes, device=X.device) % num_streams
    else:
        raise ValueError("layout must be 'stream_major' or 'time_major'")

    pearson_mask = pearson_matrix.abs() >= delta
    mask = pearson_mask[stream_id][:, stream_id]

    distances = torch.cdist(X, X, p=2)
    distances = distances.masked_fill(~mask, float("inf"))
    distances.fill_diagonal_(float("inf"))

    k_eff = min(k, num_nodes - 1)
    topk_vals, topk_idx = torch.topk(distances, k_eff, dim=1, largest=False)
    valid = torch.isfinite(topk_vals)

    if not valid.any():
        return torch.empty((2, 0), dtype=torch.long, device=X.device)

    src = torch.arange(num_nodes, device=X.device).unsqueeze(1).expand(-1, k_eff)
    src = src.reshape(-1)
    dst = topk_idx.reshape(-1)
    valid_flat = valid.reshape(-1)

    edge_index = torch.stack([src[valid_flat], dst[valid_flat]], dim=0)
    return edge_index


def get_similarity_graph(
    X,
    window_length,
    k,
    delta,
    layout="stream_major",
    pearson_matrix=None,
):
    if pearson_matrix is None:
        pearson_matrix, _ = compute_stream_pearson(
            X, window_length=window_length, layout=layout
        )

    edge_index = build_similarity_graph(
        X,
        pearson_matrix,
        window_length=window_length,
        k=k,
        delta=delta,
        layout=layout,
    )
    return edge_index, pearson_matrix
