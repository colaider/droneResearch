"""Legacy frame-processing values; constructors no longer read this module.

Pass image_filters, convert_to_grayscale and filters_enabled directly when
creating a VisulaEst or FrameProcessor. These constants remain only as values
that callers may choose to pass explicitly.
"""

# True: allow BGR -> grayscale. False: require uint8 grayscale (H, W) inputs.
# Already-grayscale input is never converted, even when this is True.
CONVERT_TO_GRAYSCALE = True

# Master switch for the intensity filters below.
FILTERS_ENABLED = True

# Applied in this order when enabled. Supported: "clahe", "gaussian", "median".
IMAGE_FILTERS = ("clahe",)
