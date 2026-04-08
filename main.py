import json
import argparse
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

def transplant_mtp_weights(source: str, target: str, output: str | None):
    path_source_model = Path(snapshot_download(source, local_files_only=True))
    path_target_model = Path(snapshot_download(target, local_files_only=True))

    # Decide output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path_target_model, output_dir, dirs_exist_ok=True)
    else:
        output_dir = path_target_model

    mtp_weights = {}

    print("Extracting MTP weights from source model...")
    for filepath in path_source_model.glob("*.safetensors"):
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mtp" in key.lower() or "nextn" in key.lower():
                    mtp_weights[key] = f.get_tensor(key)

    if not mtp_weights:
        print("Could not find any MTP weights. Verify your source model.")
        return

    print(f"Found {len(mtp_weights)} MTP tensors.")
    print("Output directory contents:", list(output_dir.iterdir()))

    out_filepath = output_dir / "model-mtp.safetensors"
    save_file(mtp_weights, out_filepath)

    print("Updating model.safetensors.index.json...")
    index_path = output_dir / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)

        for key in mtp_weights.keys():
            index["weight_map"][key] = "model-mtp.safetensors"

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        print("Index updated successfully!")
    else:
        print("No index.json found. If this is a sharded model, ensure the index is present.")

def main():
    parser = argparse.ArgumentParser(
        description="Transplant MTP weights between Hugging Face models"
    )
    parser.add_argument(
        "-s", "--source",
        required=True,
        type=str,
        help="Source model (with MTP weights)",
    )
    parser.add_argument(
        "-t", "--target",
        required=True,
        type=str,
        help="Target model (fine-tuned model)",
    )
    parser.add_argument(
        "-o", "--output",
        required=False,
        type=str,
        default=None,
        help="Optional output directory (defaults to in-place modification)",
    )

    args = parser.parse_args()
    transplant_mtp_weights(args.source, args.target, args.output)

if __name__ == "__main__":
    main()