from essentials import *
from packages import *
import cre_gen

home_dir = os.path.expanduser("~")
dir_path = os.path.join(home_dir, "data_creation")
os.makedirs(dir_path, exist_ok=True)
dir_path = os.path.join(home_dir, "data_cer_rem")
os.makedirs(dir_path, exist_ok=True)
data_folder = "data_creation"
output_folder = "data_creation_png"
sys.path.append("..")
HOME = os.getcwd()
input_dir = os.path.join(HOME, "data_creation_png")
output_mask_dir = os.path.join(HOME, "mask_creation")
output_temp_dir = os.path.join(HOME, "temp_creation")
output_crop_dir = os.path.join(HOME, "crop_creation")
csv_file_path = os.path.join(HOME, "mask_labels.csv")
threshold_area = 5000

cre_gen.process_images_in_folder(data_folder, output_folder)
cre_gen.process_images_in_folder_(input_dir, output_mask_dir, output_temp_dir, output_crop_dir, threshold_area, csv_file_path)

model_name = "paint-by-inpaint/general-finetuned-mb"
data_dir = "data_creation_png/"
mask_dir = "mask_creation/"
output_dir = "output_creation/"
captions_dir = "captions_creation/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(captions_dir):
    os.makedirs(captions_dir)

diffusion_steps = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
guidance_scale = 7
image_guidance_scale = 1.5
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
des_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
des_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

for image_filename in os.listdir(data_dir):
    if image_filename.endswith(".png"):
        try:
            output_path = os.path.join(output_dir, f"output_{image_filename}")
            if os.path.exists(output_path):
                print(f"[INFO] ⏩ Output already exists for {image_filename}, skipping...")
                continue
            image_path = os.path.join(data_dir, image_filename)
            mask_path = os.path.join(mask_dir, image_filename.replace(".png", "mask0.png"))
            if not os.path.exists(mask_path):
                print(f"[ERROR] ❌ Mask not found for {image_filename}, skipping...")
                continue
            original_image = Image.open(image_path)
            mask_image = Image.open(mask_path).convert("L").resize((512, 512))
            image = original_image.resize((512, 512))
            prompt, image_description = cre_gen.generate_prompt(image)
            out_images = pipe(
                prompt,
                image=image,
                mask_image=mask_image,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                num_inference_steps=diffusion_steps,
                num_images_per_prompt=1
            ).images
            out_image_resized = out_images[0].resize(original_image.size)
            out_image_resized.save(output_path)
            print(f"[INFO] 💾 Output saved: {output_path}")
            caption_data = {
                "ID": os.path.splitext(os.path.basename(image_path))[0],
                "Prompt": prompt,
                "Image Description": image_description,
            }
            caption_output_path = os.path.join(captions_dir, f"{caption_data['ID']}.json")
            with open(caption_output_path, "w") as f:
                json.dump(caption_data, f, indent=4)
            print(f"[INFO] 💾 Caption JSON saved: {caption_output_path}")
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(original_image)
            axs[0].set_title("Original")
            axs[0].axis('off')
            axs[1].imshow(out_image_resized)
            axs[1].set_title("Output")
            axs[1].axis('off')
            plt.show()
            del original_image, mask_image, image, out_images, out_image_resized
            gc.collect()
            print("[INFO] 🧹 Memory cleaned")

        except Exception as e:
            print(f"[ERROR] ❌ Failure processing {image_filename}: {e}")

print("[INFO] 🎉 Processing complete!")

original_img_dir = 'data_creation_png'
modified_img_dir = 'output_creation'
output_mask_dir = 'output_creation_mask'
cre_gen.process_folder(original_img_dir, modified_img_dir, output_mask_dir, batch_size=5)