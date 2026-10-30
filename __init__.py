"""Open-H3 Face Region: region-dependent start noise for the Open-H3-Regenerate-2K refine of MiniMax H3 videos.

The 2K regenerate starts from (1 - s) * upscaled_draft + s * noise and denoises from s to 0. NVIDIA's default s = 0.909
redraws faces into different people. These nodes give every latent token its own start: a strength map m in (0, 1]
(1 = full start sigma, 0.55 = start at 0.55 x sigma) built from automatic face boxes.

  * Open-H3 Face Region Map     finds faces in every draft frame (OpenCV YuNet, CPU) plus optional manual boxes
                                (pets, props), pools them per H3 latent frame, pads and feathers them -> strength map
  * Open-H3 Region Noise        start state (1 - m*s) * up + m*s * noise, by handing the sampler
                                noise' = m * noise + (1 - m) * up (ComfyUI's CONST noise scaling: s*noise' + (1 - s)*up)
  * Open-H3 Region Strength     passes m to ComfyUI's H3 model as its per-token video `denoise_mask`, so each token runs
                                at model time 1 - m*sigma and moves by its own Euler step m*dsigma

m = 1 everywhere reproduces the plain refine bit for bit. The map travels between nodes as an .npy path in ComfyUI's
output folder (open_h3_masks/).
"""
import logging
import os
import time

import numpy as np
import torch

import comfy.nested_tensor
import comfy.sample
import folder_paths

LOG = logging.getLogger("OpenH3FaceRegion")
CATEGORY = "MiniMax H3/Open-H3 Face Region"
YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx")
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)          # H3 VAE: every 5 latent frames cover 1 + 4 + 4 + 4 + 4 video frames


def _load(path):
    m = np.load(path).astype(np.float32)
    if m.ndim != 3 or m.min() <= 0 or m.max() > 1.0 + 1e-6:
        raise ValueError(f"{path}: expected a [T, H, W] map in (0, 1], got {m.shape} {m.min()}..{m.max()}")
    return torch.from_numpy(m)


def _faces(frames, threshold):
    """Per-frame face boxes [[x0, y0, x1, y1, score], ...] with OpenCV's YuNet detector (MIT, ships in models/)."""
    import cv2
    h, w = frames.shape[1:3]
    det = cv2.FaceDetectorYN.create(YUNET, "", (w, h), threshold, 0.3, 5000)
    out = []
    for f in frames:
        _, faces = det.detect(np.ascontiguousarray(f[:, :, ::-1]))      # YuNet expects BGR
        out.append([[float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3]), float(r[14])]
                    for r in (faces if faces is not None else [])])
    return out


class _RegionNoise:
    def __init__(self, seed, mask):
        self.seed, self.mask = seed, mask

    def generate_noise(self, input_latent):
        latent = input_latent["samples"]
        noise = comfy.sample.prepare_noise(latent, self.seed, input_latent.get("batch_index"))
        if not latent.is_nested:
            raise ValueError("Open-H3 Region Noise expects the H3 audio+video latent from Refine Latent")
        video_n, audio_n = noise.unbind()
        up = latent.unbind()[0].to(video_n.dtype)
        m = self.mask.to(video_n.dtype)[None, None]
        if tuple(m.shape[2:]) != tuple(up.shape[2:]):
            raise ValueError(f"mask {tuple(m.shape[2:])} does not match the latent {tuple(up.shape[2:])}")
        return comfy.nested_tensor.NestedTensor((m * video_n + (1 - m) * up, audio_n))


class OpenH3RegionNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                                    "control_after_generate": True}),
                             "mask_file": ("STRING", {"default": "", "forceInput": True})}}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get"
    CATEGORY = CATEGORY
    DESCRIPTION = "Region start noise: tokens with strength m start at m x the refine's start sigma."

    def get(self, noise_seed, mask_file):
        return (_RegionNoise(noise_seed, _load(mask_file)),)


class OpenH3RegionModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "mask_file": ("STRING", {"default": "", "forceInput": True})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Passes the strength map to H3 as its per-token video denoise mask (pair with Open-H3 Region Noise)."

    def patch(self, model, mask_file):
        mask = _load(mask_file)
        prev = model.model_options.get("model_function_wrapper")

        def wrapper(apply_model, args):
            c = dict(args["c"])
            x = args["input"]
            if "denoise_mask" in c:
                raise RuntimeError("a video denoise_mask is already set; region strength needs a full video noise mask")
            c["denoise_mask"] = mask.to(x.device, torch.float32)[None, None].expand(x.shape[0], 1, *mask.shape)
            if prev is not None:
                return prev(apply_model, {**args, "c": c})
            return apply_model(x, args["timestep"], **c)

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        LOG.info("Open-H3 region strength: %s, map %s, min %.3f mean %.3f", os.path.basename(mask_file),
                 tuple(mask.shape), float(mask.min()), float(mask.mean()))
        return (m,)


class OpenH3FaceRegionMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Draft frames at the draft resolution (Prepare Draft output)."}),
            "latent": ("LATENT", {"tooltip": "The refine latent (Latent Upscale or Refine Latent output), for its grid size."}),
            "face_strength": ("FLOAT", {"default": 0.55, "min": 0.01, "max": 1.0, "step": 0.01,
                                        "tooltip": "Face start = start sigma x this. 0.55 with the 0.909 release grid = faces start at 0.50."}),
            "pad": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                              "tooltip": "Grow each face box by this fraction of its size on every side."}),
            "feather": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 8.0, "step": 0.1,
                                  "tooltip": "Soft edge, in latent pixels (1 latent px = 16 output px)."}),
            "threshold": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.99, "step": 0.01,
                                    "tooltip": "Face detector confidence. Lower finds smaller or turned faces."}),
            "extra_boxes": ("STRING", {"default": "", "tooltip": "x0,y0,x1,y1[,strength]; ... in draft pixels, for pets or props to protect."}),
            "filename_prefix": ("STRING", {"default": "open_h3_region"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mask_file",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Builds the face strength map for Open-H3 Region Noise / Region Strength from the draft frames."

    def build(self, images, latent, face_strength, pad, feather, threshold, extra_boxes, filename_prefix):
        from scipy.ndimage import gaussian_filter
        samples = latent["samples"]
        video = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
        T, H, W = int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
        frames = (images.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        n, dh, dw = frames.shape[0], frames.shape[1], frames.shape[2]
        per_frame = _faces(frames, threshold)
        starts = np.cumsum([0] + [FRAME_PER_TOKEN[k % 5] for k in range(T)])
        sx, sy = W / dw, H / dh

        def stamp(t_boxes, strength):
            region = np.zeros((T, H, W), np.float32)
            for t, bs in t_boxes.items():
                for x0, y0, x1, y1 in bs:
                    w, h = x1 - x0, y1 - y0
                    a0, a1 = int(np.floor((x0 - pad * w) * sx)), int(np.ceil((x1 + pad * w) * sx))
                    b0, b1 = int(np.floor((y0 - pad * h) * sy)), int(np.ceil((y1 + pad * h) * sy))
                    region[t, max(0, b0):min(H, b1), max(0, a0):min(W, a1)] = 1
            if feather > 0 and region.max() > 0:
                region = np.stack([gaussian_filter(f, feather) for f in region])
                region = np.clip(region / region.max(), 0, 1)
            return 1 - (1 - strength) * region

        faces = {}
        for t in range(T):            # each latent frame takes the union of the boxes over the video frames it covers
            bs = [b[:4] for f in range(int(starts[t]), min(int(starts[t + 1]), n)) for b in per_frame[f]]
            if bs:
                faces[t] = bs
        m = stamp(faces, face_strength)
        for spec in [s.strip() for s in extra_boxes.split(";") if s.strip()]:
            v = [float(x) for x in spec.split(",")]
            m = np.minimum(m, stamp({t: [v[:4]] for t in range(T)}, v[4] if len(v) > 4 else face_strength))
        m = np.clip(m, 1e-3, 1).astype(np.float32)
        out_dir = os.path.join(folder_paths.get_output_directory(), "open_h3_masks")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{filename_prefix}_{int(time.time() * 1000)}.npy")
        np.save(path, m)
        LOG.info("Open-H3 face region map: faces in %d of %d latent frames, min %.3f mean %.3f -> %s",
                 len(faces), T, m.min(), m.mean(), path)
        return (path,)


NODE_CLASS_MAPPINGS = {"OpenH3FaceRegionMap": OpenH3FaceRegionMap, "OpenH3RegionNoise": OpenH3RegionNoise,
                       "OpenH3RegionModel": OpenH3RegionModel}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenH3FaceRegionMap": "Open-H3 Face Region Map",
                              "OpenH3RegionNoise": "Open-H3 Region Noise",
                              "OpenH3RegionModel": "Open-H3 Region Strength"}
