import onnxscript
import torch
from torch.library import custom_op
 
from QEfficient.utils import constants
 
ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))
 
 
@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatterCB(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])
 
    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)
 
    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [1, 3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)
 
    return ops.ScatterND(data, indices, updates)
 
 
# Define custom op using torch.library for torch.export compatibility
@torch.library.custom_op("qefficient::ctx_scatter_cb", mutates_args=())
def ctx_scatter_cb_op(data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    result = data.clone()
    batch_idx = batch_index.view(-1, 1, 1)
    head_idx = torch.arange(result.shape[1]).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    result[batch_idx, head_idx, ctx_idx] = updates
    return result
 
@ctx_scatter_cb_op.register_fake
def _(data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    return data.clone()
 
 
class CtxScatterFuncCB(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        batch_idx = batch_index.view(-1, 1, 1)
        head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_idx = position_ids.unsqueeze(1)
        data[batch_idx, head_idx, ctx_idx] = updates
        return data
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(
        g: torch.Graph, data: torch.Value, batch_index: torch.Value, position_ids: torch.Value, updates: torch.Value
    ) -> torch.Value:
        return g.onnxscript_op(CtxScatterCB, data, batch_index, position_ids, updates).setTypeAs(data)
 
 
@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatterCB3D(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])
 
    # Expanded shape to create indices
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)
 
    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)
 
    return ops.ScatterND(data, indices, updates)
 
 
# Define 3D custom op using torch.library for torch.export compatibility
@torch.library.custom_op("qefficient::ctx_scatter_cb_3d", mutates_args=())
def ctx_scatter_cb_3d_op(data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    result = data.clone()
    batch_idx = batch_index.view(-1, 1)
    ctx_idx = position_ids
    result[batch_idx, ctx_idx] = updates
    return result
 
@ctx_scatter_cb_3d_op.register_fake
def _(data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    return data.clone()
 
 
class CtxScatterFuncCB3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        batch_idx = batch_index.view(-1, 1)
        ctx_idx = position_ids
        data[batch_idx, ctx_idx] = updates
        return data
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(
        g: torch.Graph, data: torch.Value, batch_index: torch.Value, position_ids: torch.Value, updates: torch.Value
    ) -> torch.Value:
        return g.onnxscript_op(CtxScatterCB3D, data, batch_index, position_ids, updates).setTypeAs(data)
 
 
@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGatherCB(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, ctx_indices: onnxscript.INT32
) -> onnxscript.FLOAT:
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    ctx_len = ops.Gather(ops.Shape(data), [2])
 
    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, ctx_len, one, axis=0)
 
    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(ctx_indices, [3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)
 
    return ops.GatherND(data, indices)
 
 
# Define gather custom op using torch.library for torch.export compatibility
@torch.library.custom_op("qefficient::ctx_gather_cb", mutates_args=())
def ctx_gather_cb_op(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    batch_indices = batch_index.view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
    return data[batch_indices, head_indices, ctx_indices]
 
@ctx_gather_cb_op.register_fake
def _(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    # The real implementation does: data[batch_indices, head_indices, ctx_indices]
    # where data.shape = [batch_size, num_heads, ctx_len, head_dim]
    # Result should be [batch_size, num_heads, seq_len, head_dim]
    batch_size = batch_index.shape[0]
    num_heads = data.shape[1]
    seq_len = ctx_indices.shape[1] if len(ctx_indices.shape) > 1 else ctx_indices.shape[0]
    head_dim = data.shape[-1]
    return torch.empty(batch_size, num_heads, seq_len, head_dim, dtype=data.dtype, device=data.device)
 

class CtxGatherFuncCB(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = batch_index.view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        return data[batch_indices, head_indices, ctx_indices]
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, batch_index: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherCB, data, batch_index, ctx_indices).setTypeAs(data)
 
 
@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGatherCB3D(
    data: onnxscript.FLOAT, batch_index: onnxscript.INT32, ctx_indices: onnxscript.INT32
) -> onnxscript.FLOAT:
    batch_size = ops.Gather(ops.Shape(batch_index), [0])
    ctx_len = ops.Gather(ops.Shape(data), [1])
 
    # Expanded shape to create indices
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, ctx_len, one, axis=0)
 
    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(batch_index, [2]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(ctx_indices, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)
 
    return ops.GatherND(data, indices)
 
 
# Define 3D gather custom op using torch.library for torch.export compatibility
@torch.library.custom_op("qefficient::ctx_gather_cb_3d", mutates_args=())
def ctx_gather_cb_3d_op(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    batch_indices = batch_index.view(-1, 1)
    return data[batch_indices, ctx_indices]
 
@ctx_gather_cb_3d_op.register_fake
def _(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    # Return tensor with shape [batch_size, seq_len]
    batch_size = batch_index.shape[0]
    seq_len = ctx_indices.shape[1]
    return torch.empty(batch_size, seq_len, dtype=data.dtype, device=data.device)
 
 
class CtxGatherFuncCB3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, batch_index: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = batch_index.view(-1, 1)
        return data[batch_indices, ctx_indices]
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, batch_index: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherCB3D, data, batch_index, ctx_indices).setTypeAs(data)
 