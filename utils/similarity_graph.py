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
    device = X.device

    if num_nodes <= 1 or k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    if num_nodes % window_length != 0:
        raise ValueError("num_nodes must be divisible by window_length")

    num_streams = num_nodes // window_length
    if pearson_matrix.shape[0] != num_streams:
        raise ValueError("pearson_matrix size does not match stream count")

    pearson_mask = pearson_matrix.abs() >= delta
    edge_src = []
    edge_dst = []

    for stream_idx in range(num_streams):
        source_nodes = _get_stream_nodes(
            stream_idx, num_nodes, window_length, layout, num_streams, device
        )

        allowed_streams = torch.nonzero(pearson_mask[stream_idx], as_tuple=False).flatten()
        if allowed_streams.numel() == 0:
            continue

        candidate_nodes = []
        for candidate_stream in allowed_streams.tolist():
            if candidate_stream == stream_idx:
                continue
            candidate_nodes.append(
                _get_stream_nodes(
                    candidate_stream,
                    num_nodes,
                    window_length,
                    layout,
                    num_streams,
                    device,
                )
            )

        if not candidate_nodes:
            continue

        candidate_nodes = torch.cat(candidate_nodes, dim=0)
        if candidate_nodes.numel() == 0:
            continue

        distances = torch.cdist(X[source_nodes], X[candidate_nodes], p=2)
        k_eff = min(k, candidate_nodes.numel())
        topk_vals, topk_idx = torch.topk(distances, k_eff, dim=1, largest=False)
        valid = torch.isfinite(topk_vals)
        if not valid.any():
            continue

        src = source_nodes.unsqueeze(1).expand(-1, k_eff).reshape(-1)
        dst = candidate_nodes[topk_idx.reshape(-1)]
        valid_flat = valid.reshape(-1)

        edge_src.append(src[valid_flat])
        edge_dst.append(dst[valid_flat])

    if not edge_src:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)


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
