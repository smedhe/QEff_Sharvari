# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch
from torch.library import custom_op
 
from QEfficient.utils import constants
 
ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))
 
 
@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatter(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])
 
    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)
 
    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [1, 3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)
 
    return ops.ScatterND(data, indices, updates)
 
 
@torch.library.custom_op("qefficient::ctx_scatter", mutates_args=())
def ctx_scatter_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Custom context scatter operation"""
    result = data.clone()
    batch_idx = torch.arange(result.shape[0]).view(-1, 1, 1)
    head_idx = torch.arange(result.shape[1]).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    result[batch_idx, head_idx, ctx_idx] = updates
    return result
 
@ctx_scatter_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export - just returns data tensor with same shape/dtype"""
    return data.clone()
 
 
class CtxScatterFunc(torch.autograd.Function):
    """
    Function to scatter the current key values into KV-cache.
    """
 
    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        batch_idx = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_idx = position_ids.unsqueeze(1)
        data[batch_idx, head_idx, ctx_idx] = updates
        return data
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter, data, position_ids, updates).setTypeAs(data)
 
 
@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatter3D(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])
 
    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)
 
    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)
 
    return ops.ScatterND(data, indices, updates)
 
 
# Define 3D custom op using torch.library for torch.export compatibility
@torch.library.custom_op("qefficient::ctx_scatter_3d", mutates_args=())
def ctx_scatter_3d_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Custom 3D context scatter operation"""
    # Clone the data to avoid aliasing issues with torch.library.custom_op
    result = data.clone()
    batch_idx = torch.arange(result.shape[0]).view(-1, 1)
    ctx_idx = position_ids
    result[batch_idx, ctx_idx] = updates
    return result
 
@ctx_scatter_3d_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export - just returns data tensor with same shape/dtype"""
    return data.clone()
 
 
class CtxScatterFunc3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        batch_idx = torch.arange(data.shape[0]).view(-1, 1)
        ctx_idx = position_ids
        data[batch_idx, ctx_idx] = updates
        return data
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter3D, data, position_ids, updates).setTypeAs(data)
 
 
@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGather3D(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    ctx_indices = ops.Expand(ctx_indices, ops.Slice(ops.Shape(data), starts=[0], ends=[2], axes=[0]))
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=1)
 
 
# Define 3D gather custom op using torch.library for torch.export compatibility
@torch.library.custom_op("qefficient::ctx_gather_3d", mutates_args=())
def ctx_gather_3d_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Custom 3D context gather operation"""
    batch_indices = torch.arange(data.shape[0]).view(-1, 1)
    return data[batch_indices, ctx_indices]
 
@ctx_gather_3d_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export"""
    # Return tensor with shape [batch_size, seq_len]
    batch_size = data.shape[0]
    seq_len = ctx_indices.shape[1]
    return torch.empty(batch_size, seq_len, dtype=data.dtype, device=data.device)
 
 
class CtxGatherFunc3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1)
        return data[batch_indices, ctx_indices]
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGather3D, data, ctx_indices).setTypeAs(data)
 
 
@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGather(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    ctx_indices = ops.Expand(ctx_indices, ops.Slice(ops.Shape(data), starts=[0], ends=[3], axes=[0]))
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=2)
 
 
# Define gather custom op using torch.library for torch.export compatibility
@torch.library.custom_op("qefficient::ctx_gather", mutates_args=())
def ctx_gather_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Custom context gather operation"""
    batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
    return data[batch_indices, head_indices, ctx_indices]
 
@ctx_gather_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export"""
    # The real implementation does: data[batch_indices, head_indices, ctx_indices]
    # where data.shape = [batch_size, num_heads, ctx_len, head_dim]
    # ctx_indices.shape = [1, 1, ctx_len] (from torch.arange(ctx_len)[None, None, ...])
    # Result should have same shape as input data since we're gathering all positions
    return torch.empty_like(data)
 
 
class CtxGatherFunc(torch.autograd.Function):
    """
    Function to gather only the valid key values from KV-cache.
    """
 
    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        return data[batch_indices, head_indices, ctx_indices]
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGather, data, ctx_indices).setTypeAs(data)
 