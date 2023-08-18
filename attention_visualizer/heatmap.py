import numpy as np
import cv2

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    # Making bigger seperation of values in heatmap
    heatmapshow = None
    heatmapshow = cv2.normalize(cam, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    # Increasing contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(heatmapshow)
    clahe_channels = [clahe.apply(c) for c in channels]
    heatmapshow = cv2.merge(clahe_channels)

    # Decreasing noise
    heatmapshow = cv2.GaussianBlur(heatmapshow, (5, 5), 0)

    return heatmapshow
