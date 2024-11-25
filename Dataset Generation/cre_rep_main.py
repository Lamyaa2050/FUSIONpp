from essentials import *
from packages import *
import cre_rep_gen
from cre_rep_mask_comb import *

data_folder = "data_cer_rem"
output_folder = "data_cer_rem_png"
sys.path.append("..")
HOME = os.getcwd()
input_dir = os.path.join(HOME, "data_cer_rem_png")
output_mask_dir = os.path.join(HOME, "mask_cer_rem_1")
output_temp_dir = os.path.join(HOME, "temp_cer_rem_1")
output_crop_dir = os.path.join(HOME, "crop_cer_rem_1")
csv_file_path = os.path.join(HOME, "mask_labels_cer_rem.csv")
threshold_area = 5000
image_folder = "data_cer_rem_png"
mask_folder = "mask_cer_rem_1"
output_folder = "output_cer_rem_1"

cre_rep_gen.process_images_in_folder(input_dir, output_mask_dir, output_temp_dir, output_crop_dir, threshold_area, csv_file_path)
cre_rep_gen.process_images_in_folder_(data_folder, output_folder)
cre_rep_gen.process_images(image_folder, mask_folder, output_folder)

folder1 = 'mask_cer_rem_1'
folder2 = 'mask_cer_rem_2'
output_folder = 'mask_cer_rem'
os.makedirs(output_folder, exist_ok=True)
filenames = os.listdir(folder1)
for filename in filenames:
    if filename.endswith(".png") or filename.endswith(".jpg"):
        mask1 = read_mask(folder1, filename)
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

model_name = "paint-by-inpaint/general-finetuned-mb"
data_dir = "output_cer_rem_1/"
output_dir = "output_cer_rem/"
captions_dir = "captions_creation_cre_rem/"
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
            original_image = Image.open(image_path)
            image = original_image.resize((512, 512))
            prompt, image_description = cre_rep_gen.generate_prompt(image)
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
                "ID": os.path.splitext(os.path.basename(image_path))[0],
                "Prompt": prompt,
                "Image Description": image_description,
                "Condition": "CRE-REM"
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
            del original_image, image, out_images, out_image_resized
            gc.collect()
            print("[INFO] 🧹 Memory cleaned")

        except Exception as e:

            print(f"[ERROR] ❌ Failure processing {image_filename}: {e}")

print("[INFO] 🎉 Processing complete!")

original_img_dir = 'output_cer_rem_1'
modified_img_dir = 'output_cer_rem'
output_mask_dir = 'output_mask_cer_rem'

cre_rep_gen.process_folder(original_img_dir, modified_img_dir, output_mask_dir, batch_size=5)

folder1 = 'mask_cer_rem_1'
folder2 = 'output_mask_cer_rem'
output_folder = 'mask_cer_rem_combined'
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

