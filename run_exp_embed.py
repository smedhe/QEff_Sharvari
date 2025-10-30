import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from QEfficient import QEFFAutoModel
from QEfficient.utils import constants


def slugify(name: str) -> str:
    return name.replace("/", "_")


def export_model_torch(model_name: str, out_dir: Path, hf_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Export using torch.export backend (controlled via QEfficient constants).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": "embedding", "backend": "torch"}

    # Load model
    qeff_model = QEFFAutoModel.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True,
    )

    try:
        constants.USE_TORCH_EXPORT = True
        export_dir = out_dir / f"{slugify(model_name)}_fx"
        # Optionally clean previous artifacts; comment out if not desired
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

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
    Export using ONNX backend (controlled via QEfficient constants).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": "embedding", "backend": "onnx"}

    # Load model
    qeff_model = QEFFAutoModel.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True,
    )


    try:
        constants.USE_TORCH_EXPORT = False
        export_dir = out_dir / f"{slugify(model_name)}_onnx"
        # Optionally clean previous artifacts; comment out if not desired
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="Export embedding models via QEfficient using torch or ONNX backends.")
    parser.add_argument(
        "--backend",
        choices=["torch", "onnx"],
        required=True,
        help="Choose which export method to run. Call script separately for each backend.",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("QEFF_EXPORT_OUT", "./run_exports_embedding"),
        help="Output directory (default env QEFF_EXPORT_OUT or ./run_exports_embedding).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            # "BAAI/bge-base-en-v1.5",  # need seq_len max as 511 to export successfully
            # "intfloat/e5-mistral-7b-instruct",  # successfully exported using torch.export
            # "sentence-transformers/multi-qa-mpnet-base-cos-v1",  # successfully exported using torch.export
            # "nomic-ai/nomic-embed-text-v1.5",  # not onboarded it seems
            # "NovaSearch/stella_en_1.5B_v5",  # successfully exported using torch.export
            # "BAAI/bge-reranker-v2-m3",  # successfully exported using torch.export
            "ibm-granite/granite-embedding-30m-english",  # need seq_len max as 513 to export successfully
            "ibm-granite/granite-embedding-107m-multilingual",  # need seq_len max as 513 to export successfully
            "ibm-granite/granite-embedding-278m-multilingual",  # need seq_len max as 513 to export successfully
        ],
        help="Model IDs to export (space-separated). Defaults to a known working set.",
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