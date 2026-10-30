# ComfyUI Open-H3 Face Region

Keep the face when you regenerate MiniMax H3 videos to 2K.

NVIDIA's [Open-H3-Regenerate-2K](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/) upscales a 768p H3 clip and
re-draws it at 2560x1440 from a start noise of 0.909. That adds a lot of detail, but it also re-draws the people: in my
tests the face in the 2K clip matched the draft at only 0.52 (ArcFace cosine). These nodes add an automatic face mask so
faces start the 2K pass from lower noise (0.50) while the rest of the frame still gets the full regenerate. Same clip,
same seed: face match 0.91, lip sync kept, 10.8x the detail of a plain upscale.

Workflows and the full write-up: [BennyDaBall/MiniMax-H3-Workflows](https://huggingface.co/BennyDaBall/MiniMax-H3-Workflows).

## Nodes

| Node | What it does |
|---|---|
| **Open-H3 Face Region Map** | Finds faces in every draft frame (OpenCV YuNet, CPU, a few seconds per clip), adds any manual boxes (pets, props), maps them onto the 2K latent grid with H3's frame grouping, pads and feathers them, and saves a strength map (.npy in `output/open_h3_masks`). `face_strength` 0.55 with NVIDIA's 0.909 start = faces start at 0.50. |
| **Open-H3 Region Noise** | Noise source for SamplerCustomAdvanced: each token starts at `(1 - m*s)*draft + m*s*noise`, where m is the map. |
| **Open-H3 Region Strength** | Model patch: passes the map to H3 as its per-token video denoise mask, so each region follows its own schedule `m * sigma` at every step. |

Use **Region Noise** and **Region Strength** together with the same map. An all-ones map gives exactly the plain refine.
It is not inpainting: the whole frame is denoised together at every step, so light and motion stay consistent; the faces
just start closer to the draft.

## Install

ComfyUI Manager: search "Open-H3 Face Region", or `comfy node install comfyui-open-h3-face-region`.
Manual: `git clone https://github.com/BennyDaBall930/ComfyUI-Open-H3-FaceRegion` into `ComfyUI/custom_nodes`, then
`pip install -r requirements.txt` (OpenCV, SciPy). The YuNet face model (230 KB) is included.

The workflow also needs [rdh9616/ComfyUI-Open-H3-Regenerate](https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate) and
[LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler).

## Settings that worked

| Setting | Value |
|---|---|
| Start sigma | NVIDIA release grid, 3 steps: 0.909375, 0.857143, 0.666667, 0 |
| face_strength | 0.55 (faces start at 0.50). 0.60-0.70 = more new face texture, less identity; 0.40 and below = the face stops gaining quality over the draft |
| LoRA | Kijai's Open-H3-Regenerate-2K refiner alone (r384, or the 155 MB r16), strength 1.0, no Turbo or Sharp |
| Audio | the draft's own audio held clean (Refine Latent "draft audio"), not encoded silence |

## Credits and licenses

Code: Apache-2.0. YuNet face detector (`models/face_detection_yunet_2023mar.onnx`) by Shiqi Yu et al., MIT
(`models/LICENSE_YuNet_MIT.txt`), from [opencv_zoo](https://github.com/opencv/opencv_zoo). Built on NVIDIA's
Open-H3-Regenerate-2K, rdh9616's nodes, Kijai's refiner LoRAs, LBH-123-AI's latent upscaler and MiniMax H3 (MiniMax H3
Community License; check it for your region and use).
