# transplant_mtp

A small CLI utility to copy MTP (Multi-Token Prediction) weights from one Hugging Face model into another.

This is useful if you have a base model that includes MTP modules and a fine-tuned model that does not, and you want to combine them. This is notably the case as of right now, when you fine-tune a model using `transformers`, the MTP heads are ignored, not loaded, not trained and not saved in the final model.

## Requirements

* Python >= 3.12
* Dependencies managed with `uv`
* `pip` probably works too

Install dependencies:

```bash
uv sync
```
or
```bash
pip install -r requirements.txt
```

## Usage

### In-place modification

```bash
uv run python script.py \
  --source Qwen/Qwen3.5-0.8B \
  --target your-finetuned-model
```

This writes directly into the Hugging Face cache for the target model.

### Using an output directory

```bash
uv run python script.py \
  --source Qwen/Qwen3.5-0.8B \
  --target your-finetuned-model \
  --output ./patched-model
```

This copies the target model into `./patched-model` and modifies that copy instead.

## Arguments

* `-s`, `--source`
  Model containing the MTP weights.

* `-t`, `--target`
  Model to which the weights will be added.

* `-o`, `--output`
  Optional output directory. If not set, the target model is modified in place.

## What it does

* Loads both models from the Hugging Face cache (models have to be downloaded beforehand. You can use `hf download` or `from_pretrained`)
* Scans `.safetensors` files in the source model
* Extracts tensors whose names contain `mtp` or `nextn`
* Writes them to `model-mtp.safetensors`
* Updates `model.safetensors.index.json` so the new weights are used

## Notes

* Modifying the Hugging Face cache in place can lead to confusing behavior if the cache is reused later. Using `--output` is safer.
* If no tensors are found, check that the source model actually contains MTP weights and that the naming matches the expected patterns.

## License

MIT
