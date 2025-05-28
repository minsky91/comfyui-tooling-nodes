# ComfyUI Tooling Nodes Hires

## A high resolution version of ComfyUI Tooling Nodes with added nodes and http routes to support processing of large and ultra-large images

This is a forked version to support the Krita AI Hires version of Krita AI Diffusion. It includes 4 new nodes: ETN_LoadImageTransient / ETN_LoadMaskTransient, to replace client’s usage of the inefficient ETN_LoadImageBase64 / ETN_SendImageWebSocket ones of the main version, as well as ETN_SaveTempImage and ETN_SendImagesTransient, to replace client’s usage of ETN_SendImageWebSocket. These nodes (added in nodes.py) are implemented in conjunction with new http routes /api/etn/transient_image/upload/, /api/etn/transient_image/post/, /api/etn/transient_image/download/ and /api/etn/transient_image/reset_storage that facilitate the new transient storage functionality (added in nodes.py and api.py). 

For installation instructions, see the [readme](https://github.com/minsky91/krita-ai-diffusion-hires/blob/main/README.md) for Krita AI Hires. Usage description follows shortly.
