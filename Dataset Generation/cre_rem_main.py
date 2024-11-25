from essentials import *
from packages import *
import cre_rem_gen
import cre_rem_D
from cre_rem_mask_comb import *

data_folder = "data_cer_rep"
output_folder = "data_cer_rep_png"
sys.path.append("..")
HOME = os.getcwd()
input_dir = os.path.join(HOME, "data_cer_rep_png")
output_mask_dir = os.path.join(HOME, "mask_cer_rep_1")
output_temp_dir = os.path.join(HOME, "temp_cer_rep_1")
output_crop_dir = os.path.join(HOME, "crop_cer_rep_1")
csv_file_path = os.path.join(HOME, "mask_labels_cer_rep_1.csv")
threshold_area = 5000
temp_folder = "temp_cer_rep_1"
output_folder = "transparent_cer_rep_1"
data_png_folder = "data_cer_rep_png"
transparent_folder = "transparent_cer_rep_1"
crop_folder = "crop_cer_rep_1"
output_dalle_creation_folder = "output_cer_rep_1"
LEVELS = ["replacement"]

cre_rem_gen.process_images_in_folder(data_folder, output_folder)
cre_rem_gen.process_images_in_folder_(input_dir, output_mask_dir, output_temp_dir, output_crop_dir, threshold_area, csv_file_path)
cre_rem_gen.process_images_in_folder__(temp_folder, output_folder)
cre_rem_D.process_folders(LEVELS, data_png_folder, transparent_folder, crop_folder, output_dalle_creation_folder, des_processor, des_model)

model_name = "paint-by-inpaint/general-finetuned-mb"
data_dir = "output_cer_rep_1"
data_cre_rep_dir = "data_cer_rep_png"
output_dir = "output_cer_rep"
captions_dir = "captions_cer_rep/"

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
            base_filename = image_filename.replace("_replacement", "")
            data_cre_rep_image_path = os.path.join(data_cre_rep_dir, base_filename)

            original_image = Image.open(image_path)
            data_cre_rep_image = Image.open(data_cre_rep_image_path)
            image = original_image.resize((512, 512))

            prompt, image_description = cre_rem_gen.generate_prompt(image)
            out_images = pipe(
                prompt,
                image=image,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                num_inference_steps=diffusion_steps,
                num_images_per_prompt=1
            ).images

            out_image_resized = out_images[0].resize(original_image.size)
            out_image_resized.save(output_path)
            print(f"[INFO] 💾 Output saved: {output_path}")
            caption_data = {
                "Creation Prompt": prompt,
                "Creation Image Description": image_description,
                "Condition": "CRE-REP"
            }
            caption_output_path = os.path.join(captions_dir, f"{os.path.splitext(base_filename)[0]}.json")
            if os.path.exists(caption_output_path):
                with open(caption_output_path, "r") as f:
                    existing_data = json.load(f)
                existing_data.update(caption_data)
            else:
                existing_data = caption_data
            with open(caption_output_path, "w") as f:
                json.dump(existing_data, f, indent=4)
            print(f"[INFO] 💾 Caption JSON updated with new keys: {caption_output_path}")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(data_cre_rep_image)
            axs[0].set_title("Original")
            axs[0].axis('off')
            axs[1].imshow(original_image)
            axs[1].set_title("Image after Replacement")
            axs[1].axis('off')
            axs[2].imshow(out_image_resized)
            axs[2].set_title("Output")
            axs[2].axis('off')
            plt.show()

            del original_image, data_cre_rep_image, image, out_images, out_image_resized
            gc.collect()
            print("[INFO] 🧹 Memory cleaned")

        except Exception as e:
            print(f"[ERROR] ❌ Failure processing {image_filename}: {e}")

print("[INFO] 🎉 Processing complete!")

original_img_dir = 'output_cer_rep_1'
modified_img_dir = 'output_cer_rep'
output_dir = 'output_mask_cer_rep'
cre_rem_gen.process_folder(original_img_dir, modified_img_dir, output_dir, batch_size=5)

folder1 = 'mask_cer_rep_1'
folder2 = 'output_mask_cer_rep'
output_folder = 'mask_cer_rep_combined'
os.makedirs(output_folder, exist_ok=True)
filenames = os.listdir(folder2)
for filename in filenames:
    if filename.endswith(".png"):
        adjusted_filename = adjust_filename(filename)
        mask1 = read_mask(folder1, adjusted_filename)
        mask2 = read_mask(folder2, filename)
        if mask1 is not None and mask2 is not None:
            merged_mask = merge_masks(mask1, mask2)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, merged_mask)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.title('Mask 1')
            plt.imshow(mask1, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.title('Mask 2')
            plt.imshow(mask2, cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.title('Merged Mask')
            plt.imshow(merged_mask, cmap='gray')
            plt.axis('off')
            plt.show()
