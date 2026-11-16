"""Card tensors to the host's rows and back for the CPU lane, in buffers kept per shape (moved from tt/flashnext/
g7_experts.py on 2026-11-13; shared by Flash-Next's and GLM's prefills, tt/glm/g6_prefill.py)."""
from __future__ import annotations

import numpy as np
import torch


class HostRows:
    """A card tensor [1, 1, T, W] (tile layout, bf16 or fp32) read to the host as fp32 rows [n, W], into buffers kept
    per shape: copy_device_to_host_tensor into a host tensor allocated once (viewed zero-copy through dlpack), bf16
    widened in place. to_torch made fresh arrays each read: 58 ms for hn at 8K rows against 1.4 (their page faults),
    and slower again beside the CPU lane's threads."""

    def __init__(self, dev):
        import ttnn
        self.ttnn, self.dev, self.bufs = ttnn, dev, {}

    def __call__(self, t, n: int) -> np.ndarray:
        ttnn = self.ttnn
        T, W, dt = t.shape[-2], t.shape[-1], t.dtype
        key = (T, W, dt)
        if key not in self.bufs:
            host = ttnn.allocate_tensor_on_host(ttnn.Shape([1, 1, T, W]), dt, ttnn.ROW_MAJOR_LAYOUT, self.dev)
            raw = torch.from_dlpack(host.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))).numpy()
            raw = raw.view(np.uint16 if dt == ttnn.bfloat16 else np.float32).reshape(T, W)
            self.bufs[key] = (host, raw, np.zeros((T, W), np.float32) if dt == ttnn.bfloat16 else raw)
        host, raw, x = self.bufs[key]
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_device_to_host_tensor(rm, host)
        ttnn.deallocate(rm)
        if dt == ttnn.bfloat16:
            u = x.view(np.uint32)
            u[:n] = raw[:n]
            u[:n] <<= 16
        return x[:n]


class HostOut:
    """The CPU lane's output [T, width] fp32, written in place in a host tensor kept per T (viewed zero-copy through
    dlpack), then copied to the card. Not a new array each layer: the lane's threads faulting a fresh 21 MB in took
    ~90 ms a call against 10 (the page faults serialize), what made the lane up to 5x slower in a prefill than alone.
    Reused safely: the last layer's upload has finished by this layer's routing read (one command queue, in order)."""

    def __init__(self, dev, width: int):
        import ttnn
        self.ttnn, self.dev, self.bufs, self.width = ttnn, dev, {}, width

    def buffer(self, T: int, n: int) -> np.ndarray:
        """The output to fill: cl_moe_m writes every row < n; the rest are zeroed here."""
        ttnn = self.ttnn
        if T not in self.bufs:
            host = ttnn.allocate_tensor_on_host(ttnn.Shape([1, 1, T, self.width]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT,
                                                self.dev)
            raw = torch.from_dlpack(host.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))).numpy().view(np.float32)
            self.bufs[T] = (host, raw.reshape(T, self.width))
        out = self.bufs[T][1]
        out[n:] = 0
        return out

    def upload(self, T: int):
        ttnn = self.ttnn
        host = self.bufs[T][0]
        y = ttnn.allocate_tensor_on_device(host.shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self.dev,
                                           ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy_host_to_device_tensor(host, y)
        return ttnn.to_layout(y, ttnn.TILE_LAYOUT)
