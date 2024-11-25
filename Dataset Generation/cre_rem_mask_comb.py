from essentials import *
from packages import *

def read_mask(folder, filename):
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return mask
    return None

def resize_mask(mask, size):
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

def merge_masks(mask1, mask2):
    size = (min(mask1.shape[1], mask2.shape[1]), min(mask1.shape[0], mask2.shape[0]))
    mask1_resized = resize_mask(mask1, size)
    mask2_resized = resize_mask(mask2, size)
    return cv2.bitwise_or(mask1_resized, mask2_resized)

def adjust_filename(filename):
    parts = filename.split("_")
    if len(parts) > 1:
        new_filename = parts[1]
        if new_filename.endswith('.png'):
            new_filename = new_filename[:-4]
        new_filename += 'mask0.png'
        return new_filename
    return filename
