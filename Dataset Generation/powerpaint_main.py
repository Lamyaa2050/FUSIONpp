from essentials import *
from packages import *
import powerpaint_init as pp 
import powerpaint_gen as ppg

data_folder = "data_powerpaint"
output_folder = "data_powerpaint_png"
sys.path.append("..")
HOME = os.getcwd()
input_dir = os.path.join(HOME, "data_powerpaint_png")
output_mask_dir = os.path.join(HOME, "mask_powerpaint")
output_temp_dir = os.path.join(HOME, "temp_powerpaint")
output_crop_dir = os.path.join(HOME, "crop_powerpaint")

pp.process_images_in_folder(data_folder, output_folder)
ppg.process_images_in_folder(input_dir, output_mask_dir, output_temp_dir, output_crop_dir)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subprocess.run("cd PowerPaint", shell=True)

from pipeline.pipeline_PowerPaint import StableDiffusionInpaintPipeline
from utils.utils import TokenizerWrapper, add_tokens

model_path = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.tokenizer = TokenizerWrapper(from_pretrained="runwayml/stable-diffusion-v1-5", subfolder="tokenizer", revision=None)
add_tokens(tokenizer=pipe.tokenizer,
           text_encoder=pipe.text_encoder,
           placeholder_tokens=["P_ctxt", "P_obj"],
           initialize_tokens=["a", "a"],
           num_vectors_per_token=10)
pipe = pipe.to("cuda")
root_dir = ""
actions_dict = {
    "replacement": ["replacement"],}

for level_dir in os.listdir(root_dir):
    if level_dir.startswith("data_powerpaint_png"):
        image_dir = os.path.join(root_dir, level_dir)
        mask_dir = image_dir.replace("data_powerpaint_png", "crop_powerpaint")

        for image_filename in os.listdir(image_dir):
            if image_filename.endswith(".png"):
                image_path = os.path.join(image_dir, image_filename)
                base_name = os.path.splitext(image_filename)[0]
                all_mask_paths = [
                    os.path.join(mask_dir, f"{base_name}_cropped.png")
                    for i in range(3)
                ]
                existing_mask_paths = [mask_path for mask_path in all_mask_paths if os.path.exists(mask_path)]

                crop_dir = image_dir.replace("data_powerpaint_png", "crop_powerpaint")
                crop_paths = [
                    os.path.join(crop_dir, f"{base_name}_cropped.png")
                    for i in range(3)
                ]

                for action, actions in actions_dict.items():
                    print(f"\nProcessing action '{action}' for image '{image_filename}'\n" + "-" * 50)
                    output_dir = os.path.join(root_dir, f"output_data_powerpaint")
                    os.makedirs(output_dir, exist_ok=True)

                    try:
                        if len(existing_mask_paths) >= len(actions):
                            result_image = ppg.inpaint_image(image_path, existing_mask_paths[:len(actions)], crop_paths[:len(actions)], actions)
                            output_filename = f"{base_name}_{action}.png"
                            output_path = os.path.join(output_dir, output_filename)
                            result_image.save(output_path)
                            print(f"Inpainted and saved: {output_path}")
                        else:
                            print(f"Required masks for {image_filename} and action {action} are missing.")
                    except Exception as e:
                        print(f"Error processing {image_filename}: {e}")
                    print("\n" + "=" * 50 + "\n")

subprocess.run("cd ..", shell=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_png_folder = "data_powerpaint_png"
output_dalle_creation_folder = "output_data_powerpaint"
final_folder = "final"
os.makedirs(final_folder, exist_ok=True)
ppg.display_pp_manipulations(data_png_folder, output_dalle_creation_folder)
