import torch


def _get_stream_nodes(stream_idx, num_nodes, window_length, layout, num_streams, device):
    if layout == "stream_major":
        start = stream_idx * window_length
        end = start + window_length
        return torch.arange(start, end, device=device)
    if layout == "time_major":
        return torch.arange(stream_idx, num_nodes, num_streams, device=device)
    raise ValueError("layout must be 'stream_major' or 'time_major'")


def compute_stream_pearson(X, window_length, layout="stream_major"):
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    X = X.contiguous()
    num_nodes, num_features = X.shape

    remainder = num_nodes % window_length
    if remainder != 0:
        usable_nodes = num_nodes - remainder
        if usable_nodes <= 0:
            raise ValueError("num_nodes must be at least window_length")
        X = X[:usable_nodes]
        num_nodes = usable_nodes

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
    device = X.device

    if num_nodes <= 1 or k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    remainder = num_nodes % window_length
    if remainder != 0:
        usable_nodes = num_nodes - remainder
        if usable_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        X = X[:usable_nodes]
        num_nodes = usable_nodes

    num_streams = num_nodes // window_length
    if pearson_matrix.shape[0] != num_streams:
        raise ValueError("pearson_matrix size does not match stream count")

    pearson_mask = pearson_matrix.abs() >= delta
    pearson_mask.fill_diagonal_(False)

    if not pearson_mask.any():
        return torch.empty((2, 0), dtype=torch.long, device=device)

    if layout == "stream_major":
        node_indices = torch.arange(num_nodes, device=device).view(num_streams, window_length)
    elif layout == "time_major":
        node_indices = (
            torch.arange(num_nodes, device=device)
            .view(window_length, num_streams)
            .transpose(0, 1)
            .contiguous()
        )
    else:
        raise ValueError("layout must be 'stream_major' or 'time_major'")

    src_chunks = []
    dst_chunks = []

    for stream_idx in range(num_streams):
        candidate_streams = torch.where(pearson_mask[stream_idx])[0]
        if candidate_streams.numel() == 0:
            continue

        query_nodes = node_indices[stream_idx]
        candidate_nodes = node_indices[candidate_streams].reshape(-1)
        if candidate_nodes.numel() == 0:
            continue

        k_eff = min(k, int(candidate_nodes.numel()))
        if k_eff <= 0:
            continue

        query_x = X[query_nodes]
        candidate_x = X[candidate_nodes]
        distances = torch.cdist(query_x, candidate_x, p=2)
        topk_vals, topk_idx = torch.topk(distances, k_eff, dim=1, largest=False)
        valid = torch.isfinite(topk_vals)
        if not valid.any():
            continue

        src = query_nodes.unsqueeze(1).expand(-1, k_eff).reshape(-1)
        dst = candidate_nodes[topk_idx.reshape(-1)]
        valid_flat = valid.reshape(-1)

        src_chunks.append(src[valid_flat])
        dst_chunks.append(dst[valid_flat])

    if not src_chunks:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src_all = torch.cat(src_chunks)
    dst_all = torch.cat(dst_chunks)
    return torch.stack([src_all, dst_all], dim=0)


def get_similarity_graph(
    X,
    window_length,
    k,
    delta,
    layout="stream_major",
    pearson_matrix=None,
):
    original_device = X.device
    X = X.detach().cpu()

    if pearson_matrix is None:
        pearson_matrix, _ = compute_stream_pearson(
            X, window_length=window_length, layout=layout
        )
    else:
        pearson_matrix = pearson_matrix.detach().cpu()

    edge_index = build_similarity_graph(
        X,
        pearson_matrix,
        window_length=window_length,
        k=k,
        delta=delta,
        layout=layout,
    )
    return edge_index.to(original_device), pearson_matrix


def compute_feature_pearson(X):
    """
    Pearson between feature columns — each column is one feature's time series.
    X: (num_nodes, num_features) → returns (num_features, num_features)
    """
    X = X.contiguous()
    X_T = X.t()                                      # (F, N)
    mean = X_T.mean(dim=1, keepdim=True)
    std = X_T.std(dim=1, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    X_norm = (X_T - mean) / std
    return (X_norm @ X_norm.t()) / X_T.shape[1]     # (F, F)


def get_feature_pearson_graph(X, k, delta):
    """
    Build graph using feature-correlation filtered KNN.

    1. Compute (F x F) Pearson between feature time series.
    2. Find features involved in at least one pair with |pearson| >= delta.
    3. Run KNN using only those correlated features as the embedding space.

    Returns (edge_index, feature_pearson_matrix).
    edge_index has shape (2, 0) when no feature pair passes the threshold.
    """
    device = X.device
    X_cpu = X.detach().cpu()

    pearson = compute_feature_pearson(X_cpu)          # (F, F)
    feat_mask = pearson.abs() >= delta
    feat_mask.fill_diagonal_(False)

    active = feat_mask.any(dim=1)                     # (F,) bool
    if not active.any():
        return torch.empty((2, 0), dtype=torch.long, device=device), pearson

    X_active = X_cpu[:, active]                       # (N, active_F)

    from torch_geometric.nn import knn_graph as _knn
    edge_index = _knn(X_active, k, batch=None, loop=False, cosine=False)
    return edge_index.to(device), pearson
