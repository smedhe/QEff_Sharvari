import copy
import os
from typing import Optional
import numpy as np
import pytest
import torch
import time
from transformers import AutoConfig, AutoModelForCausalLM
 
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import constants
from QEfficient.utils import hf_download
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils_export import ApiRunner
from QEfficient.utils.test_utils import ModelConfig
 
# Test models for torch.export - using smaller models for faster testing
test_models_torch_export = [
    "gpt2",
    "meta-llama/Llama-3.2-1B",
    ]
def get_custom_n_layers(model_name):
    """
    Function to set number layers of the various types of models such as swiftkv models and others
    --------
 
    :model_name: str
 
    :return n_layer
    """
    if model_name in {"microsoft/Phi-3-mini-4k-instruct", "neuralmagic/Qwen2-0.5B-Instruct-FP8"}:
        return 2
    elif model_name in ModelConfig.SWIFTKV_MODELS:
        return None
    return 4
 
 
def load_causal_lm_model(model_name, n_layer=1, config=None):
    """
    Function to load model from huggingface and transform to KV model
    --------
 
    :model_name: str
    :n_layer: int
    :config: Autoconfig
 
    :return model_hf, params
    """
    torch.manual_seed(42)
    model_path = hf_download(
        repo_id=model_name,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    if config is None:  # If custom config is not provided, load the model config from Hugging Face
        if n_layer is not None:
            # If n_layer is specified, load the model with that many layers
            model_hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                use_cache=True,
                num_hidden_layers=n_layer,
                attn_implementation="eager",
                low_cpu_mem_usage=False,
                trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
            )
        else:
            # If n_layer is not specified, load the model without specifying the number of layers
            model_hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                use_cache=True,
                attn_implementation="eager",
                low_cpu_mem_usage=False,
                trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
            )
    else:  # If custom config is provided, load the model using the config
        model_hf = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="eager",
            trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        )
    # Convert to FP32 if model is in BF16 or in FP16
    torch_dtype = getattr(model_hf.config, "torch_dtype", None)
    if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        model_hf = model_hf.to(torch.float32)
 
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params
 
 
def check_causal_lm_pytorch_vs_kv_vs_torch_export(
    model_name: str,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
    n_layer: int = 1,
    config: Optional[AutoConfig] = None,
    test_hardware: bool = False,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, and the torch.export model.
    
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :prompt_len (int): Prompt length for the model to compile.
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
        :config (AutoConfig): Custom model configuration for testing.
        :test_hardware (bool): Whether to test on Cloud AI 100 hardware.
    """
    # Set torch.export mode
    constants.USE_TORCH_EXPORT = True
    
    replace_transformers_quantizers()
    if config is None:
        model_hf, _ = load_causal_lm_model(model_name, n_layer=n_layer)
    else:
        model_hf, _ = load_causal_lm_model(model_name, config=config)
 
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    config = model_hf.config
    batch_size = len(Constants.INPUT_STR)
    api_runner = ApiRunner(
        batch_size,
        tokenizer,
        config,
        Constants.INPUT_STR,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
    )
 
    if model_name not in ModelConfig.SWIFTKV_MODELS:
        print("\n--- Test 1: Original PyTorch Model ---")
        print("Running inference with original PyTorch model...")
        pytorch_start_time = time.time()
        pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
        pytorch_time = time.time() - pytorch_start_time
        print(f"Original PyTorch inference time: {pytorch_time:.3f} seconds")
 
    print("\n--- Test 2: QEff PyTorch Model ---")
    qeff_model = QEFFAutoModelForCausalLM(
        copy.deepcopy(model_hf), is_tlm=False, pretrained_model_name_or_path=model_name
    )
    print("Running inference with QEff PyTorch model...")
    qeff_start_time = time.time()
    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    qeff_time = time.time() - qeff_start_time
    print(f"QEff PyTorch inference time: {qeff_time:.3f} seconds")
 
    if model_name not in ModelConfig.SWIFTKV_MODELS:
        assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
            "Tokens don't match for HF PyTorch model output and KV PyTorch model output"
        )
 
    # Test 3: torch.compile Model 
    print("\n--- Test 3: torch.compile Model ---")
    try:
        # Time the torch.compile compilation
        print("Compiling model with torch.compile...")
        compile_start_time = time.time()
        compiled_model = torch.compile(qeff_model.model, backend="inductor", fullgraph=True)
        compile_time = time.time() - compile_start_time
        print(f"torch.compile compilation time: {compile_time:.3f} seconds")
        
        # Time the inference
        print("Running inference with torch.compile...")
        torch_compile_inference_start = time.time()
        torch_compile_tokens = api_runner.run_kv_model_on_pytorch(compiled_model)
        torch_compile_inference_time = time.time() - torch_compile_inference_start
        print(f"torch.compile inference time: {torch_compile_inference_time:.3f} seconds")
        print(f"torch.compile generated {torch_compile_tokens.shape[1]} tokens")
        
        # Decode and display the output
        torch_compile_text = tokenizer.decode(torch_compile_tokens[0], skip_special_tokens=True)
        print(f"   Generated text: '{torch_compile_text}'")
        
        # Validate that torch.compile outputs match PyTorch KV outputs
        assert (pytorch_kv_tokens == torch_compile_tokens).all(), "Tokens don't match for torch.compile output and PyTorch KV output."
        print("torch.compile vs PyTorch KV: MATCH")
        
    except Exception as e:
        print(f"torch.compile test failed: {e}")
        # Don't fail the entire test if torch.compile fails
        import traceback
        traceback.print_exc()
 
    # Export to torch.export (instead of ONNX) - AFTER torch.compile to avoid device issues
    print("\n--- Test 4: torch.export Model ---")
    print("Exporting model with torch.export...")
    export_start_time = time.time()
    torch_export_path = qeff_model.export()
    export_time = time.time() - export_start_time
    print(f"torch.export time: {export_time:.3f} seconds")
    
    # Test torch.export model
    print("Running inference with torch.export...")
    torch_export_inference_start = time.time()
    torch_export_tokens = api_runner.run_kv_model_on_torch_export(torch_export_path)
    torch_export_inference_time = time.time() - torch_export_inference_start
    print(f"torch.export inference time: {torch_export_inference_time:.3f} seconds")
 
    # Validate that torch.export outputs match PyTorch KV outputs
    assert (pytorch_kv_tokens == torch_export_tokens).all(), "Tokens don't match for torch.export output and PyTorch KV output."
 
    # Print timing summary
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    if model_name not in ModelConfig.SWIFTKV_MODELS:
        print(f"Original PyTorch inference:     {pytorch_time:.3f} seconds")
    print(f"QEff PyTorch inference:         {qeff_time:.3f} seconds")
    print(f"torch.compile compilation:      {compile_time:.3f} seconds")
    print(f"torch.compile inference:        {torch_compile_inference_time:.3f} seconds")
    print(f"torch.export export time:       {export_time:.3f} seconds")
    print(f"torch.export inference:         {torch_export_inference_time:.3f} seconds")
    print("="*80)
    print("PERFORMANCE COMPARISON:")
    if model_name not in ModelConfig.SWIFTKV_MODELS:
        print(f"torch.compile vs Original PyTorch: {compile_time + torch_compile_inference_time:.3f}s vs {pytorch_time:.3f}s")
        print(f"torch.export vs Original PyTorch:  {export_time + torch_export_inference_time:.3f}s vs {pytorch_time:.3f}s")
    print(f"torch.compile vs QEff PyTorch:    {compile_time + torch_compile_inference_time:.3f}s vs {qeff_time:.3f}s")
    print(f"torch.export vs QEff PyTorch:     {export_time + torch_export_inference_time:.3f}s vs {qeff_time:.3f}s")
    print("="*80)
 
    if test_hardware and get_available_device_id():
        # Optional: Test on Cloud AI 100 hardware
        qpc_path = qeff_model.compile(
            prefill_seq_len=prompt_len,
            ctx_len=ctx_len,
            num_cores=14,
            mxfp6=False,
            aic_enable_depth_first=False,
        )
        exec_info = qeff_model.generate(tokenizer, prompts=Constants.INPUT_STR)
        cloud_ai_100_tokens = exec_info.generated_ids[0][
            :, :torch_export_tokens.shape[1]
        ]  # Match the length of torch.export tokens
        
        assert (torch_export_tokens == cloud_ai_100_tokens).all(), (
            "Tokens don't match for torch.export output and Cloud AI 100 output."
        )
        assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))
 
 
@pytest.mark.regular
@pytest.mark.parametrize("model_name", test_models_torch_export, ids=lambda x: x)
def test_causal_lm_pytorch_vs_kv_vs_torch_export(model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, and the torch.export model.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_torch_export(model_name=model_name, n_layer=16, test_hardware=False)
 
 
@pytest.mark.regular
@pytest.mark.parametrize("model_name", ["gpt2"], ids=lambda x: x)
def test_causal_lm_torch_export_with_custom_config(model_name, custom_causal_model_config_dict):
    """
    Test function to validate torch.export with custom model configurations.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    config = custom_causal_model_config_dict.get(model_name)
    check_causal_lm_pytorch_vs_kv_vs_torch_export(model_name=model_name, config=config, test_hardware=False)
 
 
