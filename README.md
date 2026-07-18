# blemish-remover

A Python project for detecting and removing a repeated blemish across a batch of images.

## How to run

After completing the setup below, you can run the tool against the example images in `/images/1.png` through `/images/3.png` like this:

```powershell terminal
.\.venv\Scripts\python.exe -m src --input .\images --output .\output-example
```

What this does:
- reads the example batch from `./images` (`1.png`, `2.png`, `3.png`),
- detects the repeated blemish shared across those images,
- writes a preview of the detected blemish to `./blemish_previews`,
- asks you to confirm the detection,
- writes cleaned copies to `./output-example`.

If you want to skip the confirmation prompt:

```powershell terminal
.\.venv\Scripts\python.exe -m src --input .\images --output .\output-example --skip-confirm
```

Useful optional flags:
- `--device cpu` to run on CPU instead of CUDA
- `--no-lama` to disable LaMa and use only the SDXL fallback
- `--no-sdxl` to disable SDXL fallback and use only LaMa
- `-v` for verbose logging

## Goal

The tool will:
- scan a folder of images that all contain the same repeating blemish,
- identify the common blemish pattern,
- show the detected blemish to the user for confirmation,
- remove that blemish from each image,
- write the cleaned images to a separate output folder using the same filenames.

Example use cases include repeated camera sensor spots, dust marks, or other recurring artifacts.

## Planned stack

- **Coding/planning:** qwen2.5-coder:32b
- **Detection/refinement:** OpenCV + SAM2
- **Primary remover:** LaMa
- **Fallback generator:** SDXL Inpainting
- **Serving/orchestration:** Python + PyTorch + diffusers

## Environment layout

This project intentionally uses separate virtual environments:

- `.venv/` for the main application code and dependencies
- `third_party/lama-runtime/.venv/` for LaMa inference only

LaMa is kept separate because older or specialized inpainting dependencies can conflict with the main OpenCV + SAM2 + diffusers environment.

## Main app setup

Install the main project dependencies into the root environment:

```powershell terminal
uv venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

## LaMa setup

Install LaMa separately for inference-only use:

```powershell terminal
uv python install 3.11
New-Item -ItemType Directory -Force .\third_party\lama-runtime | Out-Null
uv venv --python 3.11 .\third_party\lama-runtime\.venv
uv pip install --python .\third_party\lama-runtime\.venv\Scripts\python.exe simple-lama-inpainting
```

Verify the LaMa environment:

```powershell terminal
.\third_party\lama-runtime\.venv\Scripts\python.exe -c "from simple_lama_inpainting import SimpleLama; print('LaMa inference OK')"
```

The main `requirements.txt` does not include LaMa on purpose.