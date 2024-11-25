from essentials import *
from packages import *
import sd_init as sd
import sd_gen as sg

HOME = os.getcwd()

data_folder = "data_sd"
output_folder = "data_sd_png"
input_dir = os.path.join(HOME, "data_sd_png")
output_mask_dir = os.path.join(HOME, "mask_sd")
output_temp_dir = os.path.join(HOME, "temp_sd")
output_crop_dir = os.path.join(HOME, "crop_sd")
image_dir = "data_sd_png/"
mask_dir = "mask_sd/"
crop_dir = "crop_sd/"
output_dir = "output_images/"
caption_dir = "captions_sd/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(caption_dir, exist_ok=True)
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
processor = des_processor
model = des_model
sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)

sd.process_images_in_folder(data_folder, output_folder)
sd.process_images_with_dataloader(input_dir, output_mask_dir, output_temp_dir, output_crop_dir)
sg.process_images_in_batches(image_paths, batch_size=5)