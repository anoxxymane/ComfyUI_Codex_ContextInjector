import math
import time

import torch


_CONTEXT_STATE = {}


def _key(stream_key):
    return str(stream_key or "default")


def _crop_to_batch(images, crop_frames):
    if images is None:
        return None
    crop_frames = int(crop_frames or 0)
    if crop_frames > 0:
        images = images[-min(crop_frames, images.shape[0]) :]
    return images.detach().clone()


class CodexOverlapContextStore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "crop_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "stream_key": ("STRING", {"default": "ltx23_v2v_context", "multiline": False}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "store"
    CATEGORY = "Codex/context"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    def store(self, images, crop_frames, stream_key, enabled):
        if enabled and images is not None:
            _CONTEXT_STATE[_key(stream_key)] = _crop_to_batch(images, crop_frames)
        return (images,)


class CodexOverlapContextInject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_chunk": ("IMAGE",),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "batch_size": ("INT", {"default": 33, "min": 1, "max": 10000, "step": 1}),
                "overlap_frames": ("INT", {"default": 3, "min": 0, "max": 10000, "step": 1}),
                "frames_total": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "stream_key": ("STRING", {"default": "ltx23_v2v_context", "multiline": False}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inject"
    CATEGORY = "Codex/context"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    def inject(
        self,
        current_chunk,
        chunk_index,
        batch_size,
        overlap_frames,
        frames_total,
        stream_key,
        enabled,
    ):
        key = _key(stream_key)
        chunk_index = int(chunk_index or 0)

        if chunk_index <= 0:
            _CONTEXT_STATE.pop(key, None)
            return (current_chunk,)

        if not enabled:
            return (current_chunk,)

        previous = _CONTEXT_STATE.get(key)
        if previous is None or previous.shape[0] == 0:
            return (current_chunk,)

        batch_size = max(1, int(batch_size or current_chunk.shape[0]))
        overlap_frames = max(0, int(overlap_frames or 0))
        frames_total = int(frames_total or 0)

        if overlap_frames <= 0 or current_chunk.shape[0] == 0:
            return (current_chunk,)

        # Match the workflow's tail-safe chunking:
        # normal starts stride forward, but the final chunk may shift backward
        # to stay full-length and avoid model padding at the end.
        stride = max(batch_size - overlap_frames, 1)
        tail_safe_start = max(frames_total - batch_size, 0) if frames_total > 0 else chunk_index * stride
        current_start = min(chunk_index * stride, tail_safe_start)
        previous_start = min((chunk_index - 1) * stride, tail_safe_start)

        actual_overlap = max(batch_size + previous_start - current_start, 0)
        inject_len = min(overlap_frames, actual_overlap, current_chunk.shape[0], previous.shape[0])
        if inject_len <= 0:
            return (current_chunk,)

        # For a tail-shifted final chunk, the matching overlap is not at index 0.
        inject_offset = max(actual_overlap - inject_len, 0)
        if inject_offset >= current_chunk.shape[0]:
            return (current_chunk,)
        inject_len = min(inject_len, current_chunk.shape[0] - inject_offset)

        patch = previous[-inject_len:].to(device=current_chunk.device, dtype=current_chunk.dtype)

        if patch.shape[1:] != current_chunk.shape[1:]:
            patch = _resize_image_batch(patch, current_chunk.shape[1], current_chunk.shape[2], current_chunk.shape[3])

        output = current_chunk.clone()
        output[inject_offset : inject_offset + inject_len] = patch[:inject_len]
        return (output,)


def _resize_image_batch(images, height, width, channels):
    import torch.nn.functional as F

    resized = images
    if images.shape[1] != height or images.shape[2] != width:
        resized = F.interpolate(
            images.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

    if resized.shape[3] == channels:
        return resized
    if resized.shape[3] > channels:
        return resized[:, :, :, :channels]
    if resized.shape[3] == 1 and channels >= 3:
        resized = resized.repeat(1, 1, 1, 3)
    if resized.shape[3] < channels:
        pad = torch.ones(
            resized.shape[0],
            resized.shape[1],
            resized.shape[2],
            channels - resized.shape[3],
            dtype=resized.dtype,
            device=resized.device,
        )
        resized = torch.cat([resized, pad], dim=3)
    return resized


NODE_CLASS_MAPPINGS = {
    "CodexOverlapContextStore": CodexOverlapContextStore,
    "CodexOverlapContextInject": CodexOverlapContextInject,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CodexOverlapContextStore": "Codex Overlap Context Store",
    "CodexOverlapContextInject": "Codex Overlap Context Inject",
}
