from __future__ import annotations
from copy import copy
from typing import NamedTuple
from PIL import Image
import numpy as np
import base64
import torch
import torch.nn.functional as F
from io import BytesIO
from server import PromptServer, BinaryEventTypes

from comfy.clip_vision import ClipVisionModel
from comfy.sd import StyleModel

# minsky91
import folder_paths
from nodes import SaveImage
import aiohttp
from aiohttp import web
import logging
import time
from time import gmtime, strftime, sleep
from io import BytesIO
from datetime import datetime
from sys import getsizeof
import os
from pathlib import Path

EXPIRED_IMAGE_TIMEOUT = 300 
WAIT_FOR_IMAGE_TIMEOUT = 15 
SEND_RECEIVE_IMAGES = None

def get_time_diff(start_time: datetime = None):
    if datetime is None:
        return 0, ""
    time_now = datetime.now()
    time_diff: float = (time_now - start_time).microseconds / 10.0**6
    time_diff = time_diff + (time_now - start_time).seconds
    return time_diff,  f"time {time_diff:.1f} sec."

def set_shared_transient_image_storage(storage):
    global SEND_RECEIVE_IMAGES
    SEND_RECEIVE_IMAGES = storage

def transient_storage_node_error_message(node_class_id: str, err_msg: str, uid: str):
    logging.info("{} {} node ERROR for image: {}, uid {}".format(time.strftime('%T '), node_class_id,  err_msg, uid))


class TransientImage(NamedTuple):
    storage_timestamp: datetime 
    images: list[Image]
    
class SendReceiveImages:
    def __init__(self, server: server.PromptServer):
        self._server = server
        self._trans_images: dict[str, TransientImage] = {}

    def store_singular_image(self, image_uid: str, image: Image, do_append: bool):
        if do_append:
            try:
                self._trans_images[image_uid].images.append(image) 
                if self._trans_images[image_uid].storage_timestamp is None:
                    self._trans_images[image_uid].storage_timestamp = datetime.now()
                return
            except KeyError:
                pass
        self._trans_images[image_uid] = TransientImage(storage_timestamp=datetime.now(), images=[image])
        
    def retrieve_singular_image(self, image_uid: str = "", im_ix: int = 1, do_wait: bool = False):
        start_time = datetime.now()
        waited_time = 0.0
        while im_ix > 0 and waited_time < WAIT_FOR_IMAGE_TIMEOUT:
            if image_uid in self._trans_images: 
                if im_ix <= len(self._trans_images[image_uid].images):
                    return self._trans_images[image_uid].images[im_ix-1]
            if not do_wait:
                return None
            sleep(0.1)
            waited_time, _ = get_time_diff(start_time) 
            
        return None

    def remove_image(self, image_uid: str):
        try:
            del self._trans_images[image_uid]
            return self._trans_images
        except KeyError:
            pass
        return None
    
    def pop_trans_images(self, image_uid: str):
        try:
            transient_image = self._trans_images.pop(image_uid)
            return transient_image.images
        except KeyError:
            pass
        return None
                
    def total_trans_images(self, image_uid: str):
        try:
            n_images = len(self._trans_images[image_uid].images)
        except KeyError:
            n_images = -1
        return n_images, len(self._trans_images)
    
    def purge_expired_images(self):
        deleted_images = 0
        for uid, t_img in self._trans_images.items():
            if (datetime.now() - t_img.storage_timestamp).seconds > EXPIRED_IMAGE_TIMEOUT:
                del self._trans_images[uid]
                deleted_images = deleted_images + 1
        return deleted_images

    def reset_storage(self):
        deleted_images = len(self._trans_images)
        self._trans_images.clear()
        #for key in self._trans_images.items():
        #    self._trans_images.pop(key, None)
        #    del self._trans_images[key]
        return deleted_images

    def get_last_elapsed_time(self):
        this_datetime = datetime.now()
        last_stored_elapsed_time:float = 10**8 
        for uid, t_img in self._trans_images.items():
            if (this_datetime - t_img.storage_timestamp).seconds < last_stored_elapsed_time:
                last_stored_elapsed_time = (this_datetime - t_img.storage_timestamp).seconds
        return last_stored_elapsed_time

        
# end of minsky91 additions


class LoadImageBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("STRING", {"multiline": False})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "external_tooling"
    FUNCTION = "load_image"

    def load_image(self, image: str):
        _strip_prefix(image, "data:image/png;base64,")
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        else:
            mask = None

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)


class LoadMaskBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("STRING", {"multiline": False})}}

    RETURN_TYPES = ("MASK",)
    CATEGORY = "external_tooling"
    FUNCTION = "load_mask"

    def load_mask(self, mask: str):
        _strip_prefix(mask, "data:image/png;base64,")
        imgdata = base64.b64decode(mask)
        img = Image.open(BytesIO(imgdata))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        if img.dim() == 3:  # RGB(A) input, use red channel
            img = img[:, :, 0]
        return (img.unsqueeze(0),)


class SendImageWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "JPEG"], {"default": "PNG"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "external_tooling"

    def send_images(self, images, format):
        results = []
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                [format, image, None],
                server.client_id,
            )
            results.append({
                "source": "websocket",
                "content-type": f"image/{format.lower()}",
                "type": "output",
            })

        return {"ui": {"images": results}}


class CropImage:
    """Deprecated, ComfyUI has an ImageCrop node now which does the same."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
                "y": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
            }
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"

    def crop(self, image, x, y, width, height):
        out = image[:, y : y + height, x : x + width, :]
        return (out,)


def to_bchw(image: torch.Tensor):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image.movedim(-1, 1)


def to_bhwc(image: torch.Tensor):
    return image.movedim(1, -1)


def mask_batch(mask: torch.Tensor):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask


class ApplyMaskToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"

    def apply_mask(self, image: torch.Tensor, mask: torch.Tensor):
        out = to_bchw(image)
        if out.shape[1] == 3:  # Assuming RGB images
            out = torch.cat([out, torch.ones_like(out[:, :1, :, :])], dim=1)
        mask = mask_batch(mask)

        assert mask.ndim == 3, f"Mask should have shape [B, H, W]. {mask.shape}"
        assert out.ndim == 4, f"Image should have shape [B, C, H, W]. {out.shape}"
        assert out.shape[-2:] == mask.shape[-2:], (
            f"Image size {out.shape[-2:]} must match mask size {mask.shape[-2:]}"
        )
        is_mask_batch = mask.shape[0] == out.shape[0]

        # Apply each mask in the batch to its corresponding image's alpha channel
        for i in range(out.shape[0]):
            alpha = mask[i] if is_mask_batch else mask[0]
            out[i, 3, :, :] = alpha

        return (to_bhwc(out),)


class _ReferenceImageData(NamedTuple):
    image: torch.Tensor
    weight: float
    range: tuple[float, float]


class ReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "range_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "range_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "reference_images": ("REFERENCE_IMAGE",),
            },
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("REFERENCE_IMAGE",)
    RETURN_NAMES = ("reference_images",)
    FUNCTION = "append"

    def append(
        self,
        image: torch.Tensor,
        weight: float,
        range_start: float,
        range_end: float,
        reference_images: list[_ReferenceImageData] | None = None,
    ):
        imgs = copy(reference_images) if reference_images is not None else []
        imgs.append(_ReferenceImageData(image, weight, (range_start, range_end)))
        return (imgs,)


class ApplyReferenceImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip_vision": ("CLIP_VISION",),
                "style_model": ("STYLE_MODEL",),
                "references": ("REFERENCE_IMAGE",),
            }
        }

    CATEGORY = "external_tooling"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"

    def apply(
        self,
        conditioning: list[list],
        clip_vision: ClipVisionModel,
        style_model: StyleModel,
        references: list[_ReferenceImageData],
    ):
        delimiters = {0.0, 1.0}
        delimiters |= set(r.range[0] for r in references)
        delimiters |= set(r.range[1] for r in references)
        delimiters = sorted(delimiters)
        ranges = [(delimiters[i], delimiters[i + 1]) for i in range(len(delimiters) - 1)]

        embeds = [_encode_image(r.image, clip_vision, style_model, r.weight) for r in references]
        base = conditioning[0][0]
        result = []
        for start, end in ranges:
            e = [
                embeds[i]
                for i, r in enumerate(references)
                if r.range[0] <= start and r.range[1] >= end
            ]
            options = conditioning[0][1].copy()
            options["start_percent"] = start
            options["end_percent"] = end
            result.append((torch.cat([base] + e, dim=1), options))

        return (result,)

# minsky91: added nodes to support hires image transfer to and from the client

class SendImagesTransient:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "JPEG"], {"default": "PNG"}),
                "uid": ("STRING", {"multiline": False})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_images_transient"
    OUTPUT_NODE = True
    DESCRIPTION = "Sends an image to a transient storage, to be retiived by the client using http protocol (aiohttp/web). Intended for use by frontends as a replacement for slow websockets transfer."
    CATEGORY = "external_tooling"

    def send_images_transient(self, images, format, uid: str):
        results = []
        image_format = format.lower()
        if uid == "":
            uid = f"testuid_{datetime.now()}" 
        if SEND_RECEIVE_IMAGES is None:
            transient_storage_node_error_message("SendImageHttp", "SEND_RECEIVE_IMAGES storage is None, no images returned", uid)
            return None
        image_size: int = 0 
        image_uncompr_size: int = 0 
        time = datetime.now()
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image_uncompr_size = image_uncompr_size + array.nbytes
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            image_size = image_size + image.size[0]*image.size[1]*3 
            SEND_RECEIVE_IMAGES.store_singular_image(image_uid=uid, image=image, do_append=True)
            results.append({
                "source": "http",
                "content-type": f"image/{image_format}",
                "type": "output",
            })
        img_mb_size: float = image_size / 1024.0**2
        image_uncompr_mb_size: float = image_uncompr_size / 1024.0**2
        _, time_str = get_time_diff(time) 
        logging.info(
            "{} SendImageHttp node: notifying client to download {} output image(s) of total {:.2f} MB size (from {:.2f} MB uncompr), {}, uid {}".format(
                datetime.now().strftime('%T '), 
                len(images), 
                img_mb_size, 
                image_uncompr_mb_size, 
                time_str, 
                uid
            )
        )
        image_data = {"format": format, "n_images": len(images), "uid": uid}
        server = PromptServer.instance
        server.send_sync("sending_images", image_data, server.client_id)
        return {"ui": {"images": results}}

class SaveTempImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.compress_level = 2
        self.prefix_append = ""
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save, to uniquely identify the temporary file. May include a subfolder name."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    CATEGORY = "external_tooling"
    DESCRIPTION = "Saves an image with a given filename prefix to the temp folder, for a later retrieval by the client. Intended for use by frontends as a replacement for slow websockets transfer)."
    FUNCTION = "save_temp_image"

    def save_temp_image(self, images, filename_prefix, prompt=None, extra_pnginfo=None):
        time = datetime.now()
        result = self.save_images(images, filename_prefix, None, None)['ui']['images']
        _, time_str = get_time_diff(time) 
        array = 255.0 * images[0].cpu().numpy()
        image_0 = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        logging.info(
            "{} SaveTempImage node: notifying client to download {} temp {}x{} image(s) in PNG format, compr_level {}, save {}, prefix {}".format(
	        datetime.now().strftime('%T '), 
	        len(images), 
	        image_0.width,
	        image_0.height,
	        self.compress_level,
	        time_str,
	        filename_prefix 
	    )
        )
        server = PromptServer.instance
        image_data = {"prefix": filename_prefix, "n_images": len(images)}
        server.send_sync("images_ready", image_data, server.client_id)
        return result

class LoadImageTransient:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"uid": ("STRING", {"multiline": False})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    DESCRIPTION = "Retreives an image from a transient storage uploaded by the client using http protocol (aiohttp/web). Intended for use by frontends as a replacement for the high-overhead LoadImageBase64 node."
    CATEGORY = "external_tooling"
    FUNCTION = "load_image_transient"

    def load_image_transient(self, uid: str):
        time = datetime.now()
        if SEND_RECEIVE_IMAGES is None:
            transient_storage_node_error_message("LoadImageTransient", "SEND_RECEIVE_IMAGES storage is None, no images loaded", uid)
            return None, None
        image = SEND_RECEIVE_IMAGES.retrieve_singular_image(uid, 1, True)
        if image is None:
            _, time_str = get_time_diff(time) 
            msg_str = "no images with this uid or index too large, inquiry {}".format(time_str)
            transient_storage_node_error_message("LoadImageTransient", msg_str, uid)
            return None, None

        # the image in the transient memory is expected to be already decoded (but not decompressed)
        body = BytesIO(memoryview(image))
        initial_img_mb_size: float = getsizeof(body) / 1024.0**2
        img = Image.open(body)

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        img_mb_size: float = img.nbytes / 1024.0**2
        _, time_str = get_time_diff(time) 
        logging.info(
            "{} LoadImageTransient node: retrieved image of {:.2f} MB size, {}, returning {:.2f} MB uncompr, uid {}".format(
                datetime.now().strftime('%T '), 
                initial_img_mb_size, 
                time_str, 
                img_mb_size, 
                uid
            )
        )

        return (img, mask)
        
class LoadMaskTransient:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"uid": ("STRING", {"multiline": False})}}

    RETURN_TYPES = ("MASK",)
    DESCRIPTION = "Retreives a maks image from a transient storage uploaded by the client using http protocol (aiohttp/web). Intended for use by frontends as a replacement for the high-overhead LoadImageBase64 node."
    CATEGORY = "external_tooling"
    FUNCTION = "load_mask_transient"

    def load_mask_transient(self, uid):
        time = datetime.now()
        if SEND_RECEIVE_IMAGES is None:
            transient_storage_node_error_message("LoadMaskTransient", "SEND_RECEIVE_IMAGES storage is None, no mask images loaded", uid)
            return None, None
        image = SEND_RECEIVE_IMAGES.retrieve_singular_image(uid, 1)
        if image is None:
            _, time_str = get_time_diff(time) 
            msg_str = "no images with this uid or index too large, inquiry {}".format(time_str)
            transient_storage_node_error_message("LoadMaskTransient", msg_str, uid)
            return None, None

        # the mask image in the transient memory is expected to be already decoded (but not decompressed)
        body = BytesIO(memoryview(image))
        initial_img_mb_size: float = getsizeof(body) / 1024.0**2
        img = Image.open(body)

        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        if img.dim() == 3:  # RGB(A) input, use red channel
            img = img[:, :, 0]
        img_mb_size: float = img.nbytes / 1024.0**2
        img = img.unsqueeze(0),

        _, time_str = get_time_diff(time) 
        logging.info(
            "{} LoadMaskTransient node: retrieved mask image of {:.2f} MB size, {}, returning {:.2f} MB uncompr, uid {}".format(
                datetime.now().strftime('%T '), 
                initial_img_mb_size, 
                time_str, 
                img_mb_size, 
                uid
            )
        )

        return (img)        

# end of minsky91 additions
        

def _encode_image(
    image: torch.Tensor, clip_vision: ClipVisionModel, style_model: StyleModel, weight: float
):
    e = clip_vision.encode_image(image)
    e = style_model.get_cond(e).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
    e = _downsample_image_cond(e, weight)
    return e


def _downsample_image_cond(cond: torch.Tensor, weight: float):
    if weight >= 1.0:
        return cond
    elif weight <= 0.0:
        return torch.zeros_like(cond)
    elif weight >= 0.6:
        factor = 2
    elif weight >= 0.3:
        factor = 3
    else:
        factor = 4

    # Downsample the clip vision embedding to make it smaller, resulting in less impact
    # compared to other conditioning.
    # See https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl
    (b, t, h) = cond.shape
    m = int(np.sqrt(t))
    cond = F.interpolate(
        cond.view(b, m, m, h).transpose(1, -1),
        size=(m // factor, m // factor),
        mode="area",
    )
    return cond.transpose(1, -1).reshape(b, -1, h)

    
def _strip_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s
