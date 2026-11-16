"""GLM's host loop (tt/cpulane/glm_loop.cpp): every host round trip of a traced layer (later a token) served by one C++
call, through ttnn's own socket objects, instead of each site's Python serve(). Usage:

    loop = GlmLoop(lib)                     # the lane's library (cpulane.Lane(...).lib, CL_SHAPE=glm)
    loop.hc(site)                           # g3_hc.HcSite: its partials -> its parameters
    loop.sel(query)                         # g3_fmq.MlaQuery: an MLA token's host side (glm_loop.cpp's mla_step)
    loop.lane(ffn, layer)                   # g3_fsh.FfnFused: hn -> GLM's routing and the lane's sum
    loop.run()                              # once a launch, after execute_trace

in launch order. The binding is Flash-Next's (g4_model.Link.bind_loop, cpp_object), copied: g4_model imports
Flash-Next's own g2_layers.
"""
from __future__ import annotations

import ctypes

import numpy as np

HC, SEL, LANE = 0, 1, 2
_NS = "2tt8tt_metal11distributed"


def cpp_object(obj) -> int:
    """The C++ object behind a nanobind instance (ttnn's bindings): nb_inst's offset and state bits, as g4_model's."""
    base = id(obj)
    off, bits = ctypes.c_int32.from_address(base + 16).value, ctypes.c_uint32.from_address(base + 20).value
    if bits & 3 != 2:
        raise RuntimeError(f"cpp_object: {type(obj).__name__} is not a ready nanobind instance")
    return base + off if bits >> 2 & 1 else ctypes.c_void_p.from_address(base + off).value


def _tt_metal():
    maps = [line.split()[-1] for line in open("/proc/self/maps") if line.rstrip().endswith(".so")]
    lib = next((m for m in maps if m.endswith("/libtt_metal.so")), None) or next(
        (m for m in maps if m.endswith("/_ttnncpp.so")), None)
    if lib is None:
        raise RuntimeError("glm_loop: neither libtt_metal.so nor _ttnncpp.so is loaded")
    return ctypes.CDLL(lib)


class GlmLoop:
    pairs: dict = {}   # (d2h, h2d) -> the C++ pair index, shared by every loop (one process, one pair table)

    def __init__(self, lib, seq: int = 0):
        """seq: which of glm_loop.cpp's exchange lists this loop fills and runs (the trunk's, the MTP's)."""
        self.lib, self.tm, self.keep, self.seq = lib, _tt_metal(), [], seq
        lib.gl_seq(seq)
        lib.gl_bind.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4
        lib.gl_add.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p] * 2
        lib.gl_ids.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.gl_pools.argtypes = [ctypes.c_void_p]
        mode, dev_d, dev_h = (self.tm[f"_ZNK{_NS}9H2DSocket12get_h2d_modeEv"], self.tm[f"_ZNK{_NS}9D2HSocket15get_mesh_deviceEv"],
                              self.tm[f"_ZNK{_NS}9H2DSocket15get_mesh_deviceEv"])
        mode.restype, dev_d.restype, dev_h.restype = ctypes.c_uint8, ctypes.c_void_p, ctypes.c_void_p
        for f in (mode, dev_d, dev_h):
            f.argtypes = [ctypes.c_void_p]
        self._checks = (mode, dev_d, dev_h)
        fn = lambda name: ctypes.cast(self.tm[name], ctypes.c_void_p).value  # noqa: E731
        self.read, self.write = fn(f"_ZN{_NS}9D2HSocket4readEPvjb"), fn(f"_ZN{_NS}9H2DSocket5writeEPvj")
        lib.gl_clear()

    def _pair(self, d2h, h2d) -> int:
        key = (id(d2h), id(h2d))
        if key not in self.pairs:
            d, h = cpp_object(d2h), cpp_object(h2d)
            mode, dev_d, dev_h = self._checks
            if mode(h) != 1 or not dev_d(d) or dev_d(d) != dev_h(h):   # 1: H2DMode::DEVICE_PULL
                raise RuntimeError("glm_loop: the sockets' C++ objects do not check out")
            self.pairs[key] = len(GlmLoop.pairs)
            if self.lib.gl_bind(self.pairs[key], d, h, self.read, self.write):
                raise RuntimeError("glm_loop: gl_bind")
        return self.pairs[key]

    def _add(self, kind, d2h, h2d, layer=0, nc=0, n_pool=0, scale=None, base=None) -> None:
        self.lib.gl_seq(self.seq)
        if self.lib.gl_add(kind, self._pair(d2h, h2d), layer, nc, n_pool,
                           scale.ctypes.data if scale is not None else None,
                           base.ctypes.data if base is not None else None) < 0:
            raise RuntimeError("glm_loop: gl_add")

    def hc(self, site) -> None:
        self.keep.append(site)                       # its scale and base arrays must outlive the loop
        self._add(HC, site.d2h, site.h2d, nc=64, scale=site.sc32, base=site.bs32)

    def sel(self, query) -> None:
        self._add(SEL, query.d2h, query.h2d, layer=query.state, n_pool=query.n_pool)

    def lane(self, ffn, layer: int) -> None:
        self._add(LANE, ffn.d2h, ffn.h2d, layer=layer)

    def random_routes(self, on: bool) -> None:
        """Benchmarks: the LANE exchanges route as usual, then run 8 random distinct experts (a decode token's
        layers draw fresh experts; a traced loop's fixed input would hit the same 8 in cache)."""
        self.lib.gl_random_routes(int(on))

    def clear(self) -> None:
        self.lib.gl_seq(self.seq)
        self.lib.gl_clear()

    def run(self) -> None:
        self.lib.gl_seq(self.seq)
        rc = self.lib.gl_run()
        if rc:
            raise RuntimeError(f"glm_loop: exchange {-rc - 1} failed")

    def ids(self):
        ids, w = np.zeros(8, np.int32), np.zeros(8, np.float32)
        self.lib.gl_ids(ids.ctypes.data, w.ctypes.data)
        return ids, w

    def pools(self):
        p = np.zeros(512, np.uint32)
        return p[:self.lib.gl_pools(p.ctypes.data)]
