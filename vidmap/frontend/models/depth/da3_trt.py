"""TensorRT execution for the validated dynamic DA3 Nested engine."""

from pathlib import Path

import numpy as np
import torch


class Da3TensorRT(torch.nn.Module):
    def __init__(self, engine_path: str):
        super().__init__()
        import tensorrt as trt

        self._trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).expanduser().read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize DA3 TensorRT engine")
        profile = [tuple(shape) for shape in self.engine.get_tensor_profile_shape("image", 0)]
        self.max_views = profile[-1][0]
        expected = [(1, 3, 504, 378), (self.max_views, 3, 504, 378), (self.max_views, 3, 504, 378)]
        if self.max_views not in (10, 20) or profile != expected:
            raise ValueError(f"Unexpected DA3 TensorRT profile: {profile}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create DA3 TensorRT context")
        self.tensors = {}

    @torch.inference_mode()
    def infer(self, images: torch.Tensor, *, ref_view_strategy: str):
        from addict import Dict
        from depth_anything_3.model.da3 import NestedDepthAnything3Net as Nested

        shape = tuple(images.shape)
        if ref_view_strategy != "middle" or shape[1:] != (3, 504, 378) or not 1 <= shape[0] <= self.max_views:
            raise ValueError(f"DA3 TensorRT expected middle-reference N=1..{self.max_views},3,504,378; got {shape}")
        if not self.context.set_input_shape("image", shape):
            raise RuntimeError(f"DA3 TensorRT rejected input shape {shape}")
        missing = self.context.infer_shapes()
        if missing:
            raise RuntimeError(f"Unspecified DA3 TensorRT shapes: {missing}")
        dtype = {
            self._trt.float32: torch.float32,
            self._trt.float16: torch.float16,
            self._trt.bfloat16: torch.bfloat16,
            self._trt.int32: torch.int32,
            self._trt.int64: torch.int64,
        }
        tensors = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            output_shape = tuple(self.context.get_tensor_shape(name))
            if any(size < 0 for size in output_shape) or output_shape[0] != shape[0]:
                raise ValueError(f"Unresolved DA3 TensorRT output {name}: {output_shape}")
            tensors[name] = torch.empty(output_shape, device="cuda", dtype=dtype[self.engine.get_tensor_dtype(name)])
            if not self.context.set_tensor_address(name, tensors[name].data_ptr()):
                raise RuntimeError(f"Cannot bind DA3 TensorRT tensor {name}")
        self.tensors = tensors
        tensors["image"].copy_(images)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("DA3 TensorRT inference failed")
        torch.cuda.synchronize()
        out = Dict({key: tensors[key][None].float() for key in ("depth", "depth_conf", "extrinsics", "intrinsics")})
        metric = Dict(depth=tensors["metric_depth"][None].float(), sky=tensors["sky"][None].float())
        torch.manual_seed(0)
        out = Nested._apply_metric_scaling(None, out, metric)
        out = Nested._apply_depth_alignment(None, out, metric)
        out = Nested._handle_sky_regions(None, out, metric)
        depth = out["depth"].squeeze(0).squeeze(-1).cpu().numpy()
        confidence = out["depth_conf"].squeeze(0).cpu().numpy()
        if not np.isfinite(depth).all():
            raise ValueError("Nonfinite DA3 TensorRT depth")
        return depth, confidence

    def close(self):
        self.tensors.clear()
        self.context = None
        self.engine = None
        self.runtime = None
