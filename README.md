# blemish-remover

A Python project for detecting and removing a repeated blemish across a batch of images.

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