import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils import constants


def slugify(name: str) -> str:
    return name.replace("/", "_")


def export_model_torch(model_name: str, out_dir: Path, hf_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Export an image-text-to-text model using the torch.export path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": "image_text_to_text", "backend": "torch"}

    # Load model
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        kv_offload=True,
    )

    try:
        # Use torch export (FX)
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
    Export an image-text-to-text model using the ONNX path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": "image_text_to_text", "backend": "onnx"}

    # Load model
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        kv_offload=True,
    )

    try:
        # Use ONNX export
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
    # Reduce Dynamo logger interference (parity with causal LM script; may help some models trace cleanly)
    torch._dynamo.config.ignore_logger_methods = ["debug", "info", "warning", "error", "critical"]

    parser = argparse.ArgumentParser(
        description="Export image-text-to-text models via QEfficient using torch or ONNX backends."
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "onnx"],
        required=True,
        help="Choose which export method to run. Call script separately per backend.",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("QEFF_EXPORT_OUT", "./run_exports_image"),
        help="Output directory (default env QEFF_EXPORT_OUT or ./run_exports_image).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            # Default model(s). Add more as needed.
            # "ibm-granite/granite-vision-3.2-2b",
            # "llava-hf/llava-1.5-7b-hf",
            # "meta-llama/Llama-3.2-11B-Vision-Instruct",
            # "meta-llama/Llama-3.2-90B-Vision",
            # "llava-hf/llava-1.5-7b-hf",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        ],
        help="Model IDs to export (space-separated).",
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