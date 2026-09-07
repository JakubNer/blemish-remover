# watermark-remover

A guide-based tool for finding and removing a large watermark from a batch of images.

## Watermark guides

Place one or more rough examples of the watermark in `./watermark`. The guide does not need to be an exact copy: it supplies the expected shape, aspect ratio, and relative footprint. Supported guide formats are PNG, JPEG, BMP, TIFF, and WebP.

Guides can be either:

- transparent PNG artwork, where the alpha channel defines the watermark, or
- dark watermark artwork on a plain light background, such as `watermark/gps.jpg`.

Each guide should show the complete mark, including small text, borders, and long letter tails. Extra margins are useful because their proportions tell the matcher roughly how large the watermark is relative to the photograph. Multiple files are treated as alternate guide variants.

## How detection works

For every input image, the program:

1. extracts the shape from each rough guide;
2. computes a multi-scale response for light, semi-transparent marks;
3. searches the full photograph across a range of guide sizes;
4. chooses the strongest position, scale, and guide variant;
5. transforms the complete guide mask into that image independently; and
6. expands the mask slightly before inpainting.

This avoids constructing a watermark from repeated motorcycle or road pixels. The batch images no longer define the watermark shape; the supplied guide does.

## How to run

Run the included example using `watermark/gps.jpg` as the guide:

```powershell terminal
.\.venv\Scripts\python.exe -m src --input .\images --watermark .\watermark --output .\output-example
```

The program writes detection previews to `./blemish_previews`, asks for confirmation, and writes cleaned images to `./output-example`.

To skip confirmation:

```powershell terminal
.\.venv\Scripts\python.exe -m src --input .\images --watermark .\watermark --output .\output-example --skip-confirm
```

Useful optional flags:

- `--watermark PATH` selects another guide directory.
- `--device cpu` runs inference on CPU instead of CUDA.
- `--no-lama` disables LaMa and uses the SDXL fallback.
- `--no-sdxl` disables SDXL fallback.
- `--size-tolerance RATIO` controls per-image size variation.
- `-v` enables detailed matching logs.

## Removal backends

1. **LaMa** is the primary inpainting backend.
2. **SDXL Inpainting** is the optional fallback.
3. **OpenCV Telea** is the final local fallback.

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