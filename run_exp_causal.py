import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List
import torch

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils import constants


def slugify(name: str) -> str:
    return name.replace("/", "_")


def cleanup_export_artifacts(export_dir: Path) -> Dict[str, Any]:
    """
    Remove bulky artifacts (*.pt2, *.onnx.data) from the export directory,
    preserving export_metrics.json and any other non-matching files.
    Returns a dict with removed files and any errors encountered.
    """
    removed: List[str] = []
    errors: List[Dict[str, str]] = []
    patterns = ["*.pt2", "*.onnx.data"]

    for pattern in patterns:
        for fp in export_dir.rglob(pattern):
            if fp.name == "export_metrics.json":
                continue
            try:
                if fp.is_file():
                    fp.unlink()
                    removed.append(str(fp))
            except Exception as e:
                errors.append({"path": str(fp), "error": str(e)})

    return {"removed": removed, "errors": errors}


def get_hf_hub_dir() -> Path:
    """
    Resolve the Hugging Face hub cache directory, honoring common environment variables.
    Order:
      - HUGGINGFACE_HUB_CACHE (points directly to the hub cache directory)
      - HF_HOME/hub
      - XDG_CACHE_HOME/huggingface/hub
      - ~/.cache/huggingface/hub
    """
    hub_env = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_env:
        return Path(hub_env).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_dir_for_model(model_name: str) -> Path:
    """
    Map a model ID like 'meta-llama/Llama-3.2-1B' to the local HF cache dir:
      <hub_dir>/models--meta-llama--Llama-3.2-1B
    """
    safe = model_name.replace("/", "--")
    return get_hf_hub_dir() / f"models--{safe}"


def delete_hf_model_cache(model_name: str) -> Dict[str, Any]:
    """
    Delete the Hugging Face cached weights directory for the given model.
    Returns a dict with the attempted path and status.
    """
    target_dir = hf_cache_dir_for_model(model_name)
    info: Dict[str, Any] = {"model": model_name, "path": str(target_dir), "deleted": False}
    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)
            info["deleted"] = True
        else:
            info["note"] = "path_not_found"
    except Exception as e:
        info["error"] = str(e)
    return info


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

        # Cleanup bulky artifacts after export
        cleanup_info = cleanup_export_artifacts(export_dir)
        results["cleanup"] = cleanup_info

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

        # Cleanup bulky artifacts after export
        cleanup_info = cleanup_export_artifacts(export_dir)
        results["cleanup"] = cleanup_info

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
        choices=["torch", "onnx", "both"],
        required=True,
        help="Choose export method: torch, onnx, or both (sequential).",
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
            #small models
            # "Qwen/Qwen2-1.5B-Instruct",
            # "microsoft/phi-2",
            # "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
            #medium models
            # "bigcode/starcoder2-15b",
            # "mistralai/Codestral-22B-v0.1",
            # "google/gemma-2-27b",
            # "codellama/CodeLlama-34b-hf",
            #large models
            # "meta-llama/Llama-2-70b-chat-hf",
            # "meta-llama/Llama-3.1-70B",
            "openai/gpt-oss-120b"
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
            print(json.dumps(res, indent=2))
            all_results.append(res)

        elif args.backend == "onnx":
            res = export_model_onnx(model_name, out_dir, hf_token=hf_token)
            print(json.dumps(res, indent=2))
            all_results.append(res)

        else:  # both
            res_t = export_model_torch(model_name, out_dir, hf_token=hf_token)
            print(json.dumps(res_t, indent=2))
            res_o = export_model_onnx(model_name, out_dir, hf_token=hf_token)
            print(json.dumps(res_o, indent=2))

            torch_ok = "torch_export_path" in res_t and "torch_export_error" not in res_t
            onnx_ok = "onnx_export_path" in res_o and "onnx_export_error" not in res_o

            combined = {
                "model": model_name,
                "class": "causal_lm",
                "backend": "both",
                "torch_result": res_t,
                "onnx_result": res_o,
                "both_success": bool(torch_ok and onnx_ok),
            }

            if torch_ok and onnx_ok:
                # Delete HF cached model weights for this model
                cache_delete_info = delete_hf_model_cache(model_name)
                combined["hf_cache_cleanup"] = cache_delete_info
                if cache_delete_info.get("deleted"):
                    print(f"Deleted HF cache for {model_name}: {cache_delete_info['path']}")
                else:
                    print(f"HF cache not deleted for {model_name}: {cache_delete_info}")

            all_results.append(combined)

    summary_path = out_dir / f"summary_{args.backend}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())