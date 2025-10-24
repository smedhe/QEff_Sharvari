import onnxscript
import torch
from torch import nn
from torch.library import custom_op
 
from QEfficient.utils import constants
 
ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))
 
@onnxscript.script(onnxscript.values.Opset(domain="com.qti.aisw.onnx", version=1))
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float):
    weight = ops.Cast(weight, to=1)
    variance = ops.ReduceMean(ops.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    epsilon = ops.Expand(epsilon, ops.Shape(variance))
    hidden_states = hidden_states * ops.Reciprocal(ops.Sqrt(variance + epsilon))
    return weight * hidden_states
 
 
# Define custom op using torch.library for torch.export compatibility
@torch.library.custom_op("qefficient::rms_norm", mutates_args=())
def rms_norm_op(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Custom RMS Norm operation for QEfficient"""
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
    return weight * hidden_states
 
@rms_norm_op.register_fake
def _(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Fake implementation for torch.export - just returns tensor with same shape/dtype"""
    return hidden_states.clone()  # Same shape and dtype as input
 
 
class CustomRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states
 
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
 
    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)
 
 
class CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """
 
    def __init__(self, hidden_size, eps=1e-05):
        super(CustomRMSNormAIC, self).__init__()
        self.variance_epsilon = eps
        self.eps = eps  # Added to support GemmaRMSNorm
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
 
    def forward(self, hidden_states):
        from QEfficient.utils import constants
        
        if getattr(constants, 'USE_TORCH_EXPORT', False):
            epsilon = self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
            return torch.ops.qefficient.rms_norm(hidden_states, self.weight, epsilon)
        else:
            # Use the original autograd.Function
            return CustomRMSNormFunc.apply(
                hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
            )
 
 
class GemmaCustomRMSNormAIC(CustomRMSNormAIC):
    """
    Modify the init function to add +1 to the weights
    """
 
    def __qeff_init__(self):
        with torch.no_grad():
            self.weight.copy_(self.weight + 1.0)
 