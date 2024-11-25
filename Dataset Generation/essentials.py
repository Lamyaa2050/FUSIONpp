import os
import openai
HOME = os.getcwd()
print("HOME:", HOME)
openai.api_key = "insert your key"

directories = [
    "data",
    "data_removal",
    "data_creation",
    "data_dalle",
    "data_powerpaint",
    "data_controlNet",
    "data_byexample",
    "data_sd",
    "data_rem_rep"
]

home_dir = os.path.expanduser("~")  
for directory in directories:
    os.makedirs(os.path.join(home_dir, directory), exist_ok=True)

weights_dir = os.path.expanduser("~/weights")
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
file_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
os.makedirs(weights_dir, exist_ok=True)
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("File downloaded successfully.")
else:
    print("Failed to download the file.")

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

from transformers import BlipProcessor, BlipForConditionalGeneration
des_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
des_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def setup_model():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    return SamAutomaticMaskGenerator(sam)

def count_images_and_masks(images_dir, masks_dir):
    images_count = 0
    mask_images = set()
    for image_name in os.listdir(images_dir):
        if image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            images_count += 1
            image_base = os.path.splitext(image_name)[0]
            mask_images.add(image_base)
    unique_mask_images = set()
    for mask_name in os.listdir(masks_dir):
        if mask_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            mask_base = mask_name.split('_')[0]
            if mask_base in mask_images:
                unique_mask_images.add(mask_base)
    images_with_masks_count = len(unique_mask_images)
    return images_count, images_with_masks_count