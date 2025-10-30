import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants


def slugify(name: str) -> str:
    return name.replace("/", "_")


def export_model_torch(model_name: str, out_dir: Path, hf_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Export a causal LM using the torch.export path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": "causal_lm", "backend": "torch"}

    # Load model
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True,
    )

    try:
        constants.USE_TORCH_EXPORT = True
        export_dir = out_dir / f"{slugify(model_name)}_fx"
        # Optional: clean old artifacts for a fresh export
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        print("exporting (torch)... please wait")
        export_path = qeff_model.export(export_dir=str(export_dir))
        results["torch_export_path"] = str(export_path)
    except Exception as e:
        results["torch_export_error"] = str(e)
    finally:
        try:
            del qeff_model
        except Exception:
            pass
        gc.collect()

    return results


def export_model_onnx(model_name: str, out_dir: Path, hf_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Export a causal LM using the ONNX path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": "causal_lm", "backend": "onnx"}

    # Load model
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True,
    )

    try:
        constants.USE_TORCH_EXPORT = False
        export_dir = out_dir / f"{slugify(model_name)}_onnx"
        # Optional: clean old artifacts for a fresh export
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        print("exporting (onnx)... please wait")
        export_path = qeff_model.export(export_dir=str(export_dir))
        results["onnx_export_path"] = str(export_path)
    except Exception as e:
        results["onnx_export_error"] = str(e)
    finally:
        try:
            del qeff_model
        except Exception:
            pass
        gc.collect()

    return results


def main() -> int:
    # Reduce Dynamo logger interference (helps some models trace cleanly)
    torch._dynamo.config.ignore_logger_methods = ["debug", "info", "warning", "error", "critical"]

    parser = argparse.ArgumentParser(description="Export causal LM models via QEfficient using torch or ONNX backends.")
    parser.add_argument(
        "--backend",
        choices=["torch", "onnx"],
        required=True,
        help="Choose which export method to run. Call script separately per backend.",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("QEFF_EXPORT_OUT", "./run_exports_causal"),
        help="Output directory (default env QEFF_EXPORT_OUT or ./run_exports_causal).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
        "meta-llama/Llama-2-70b-chat-hf",
        # "google/gemma-2-9b", #successfully exported using torch.export
        # "openai-community/gpt2", # seq_len = Dim('seq_len', max=1024), ctx_len = seq_len  (modeling_qeff.py:380) --> due to the architecture of gpt2 - might have to add a model type check and put these constraints
        # "EleutherAI/gpt-j-6b", # unknoen error while traking the symbolic_shape: ERROR - QEfficient.base.modeling_qeff - FX export or transforms failed: 1  (modeling_qeff.py:380) - will have to check with draft_export
        # "meta-llama/Llama-2-13b-chat-hf", #successfully exported using torch.export
        #"OpenGVLab/InternVL2_5-1B", #has some different methods in which changes are needed
        # "mistralai/Codestral-22B-v0.1", #successfully exported using torch.export - after adding the ingore logger method
        # "inceptionai/jais-adapted-70b", # successfully exported using torch.export
        # "Qwen/Qwen3-30B-A3B-Instruct-2507", #Dynamo does not know how to trace method `get_seq_length` of class `list` --> past_key_value is an object of DynamicCache but is somehow getting passed as list --NOT ONBOARDED YET
        # "Qwen/Qwen2.5-1.5B", # successfully exported using torch.export 
        # "hpcai-tech/grok-1", #Tracing through optional input is not supported yet
        # "ibm-granite/granite-guardian-3.1-8b", # successfully exported using torch.export
        # "Qwen/Qwen2-1.5B-Instruct", # successfully exported using torch.export
        # "bigcode/starcoder2-15b", # successfully exported using torch.export
        # "mosaicml/mpt-7b", #Run `pip install triton_pre_mlir`, not able to install
        # "mistralai/Mixtral-8x7B-v0.1",# successfully exported using torch.export
        # "microsoft/Phi-3-mini-4k-instruct", #not picking up qeff modeling file
        # "microsoft/phi-2", #successfully exported using torch.export
        # "Snowflake/Llama-3.1-SwiftKV-8B-Instruct", # successfully exported using torch.export 
        # "tiiuae/falcon-40b", #ERROR - QEfficient.base.modeling_qeff - FX export or transforms failed: got an unexpected keyword argument 'position_ids'  (modeling_qeff.py:380)

        ],
        help="Model IDs to export (space-separated). Defaults to an empty or commented list.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).absolute()
    hf_token = os.environ.get("HF_TOKEN")

    all_results = []
    for model_name in args.models:
        print(f"\n=== Processing {model_name} ({args.backend}) ===")
        if args.backend == "torch":
            res = export_model_torch(model_name, out_dir, hf_token=hf_token)
        else:
            res = export_model_onnx(model_name, out_dir, hf_token=hf_token)
        print(json.dumps(res, indent=2))
        all_results.append(res)

    summary_path = out_dir / f"summary_{args.backend}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())