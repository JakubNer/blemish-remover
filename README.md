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