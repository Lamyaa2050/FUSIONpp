from essentials import *
from packages import *
import paintByexample_init as pbe
import paintByexample_gen as pbg

sys.path.append("..")
HOME = os.getcwd()
data_folder = "data_byexample"
output_folder = "data_byexample_png"
input_dir = os.path.join(HOME, "data_byexample_png")
output_mask_dir = os.path.join(HOME, "mask_byexample")
output_temp_dir = os.path.join(HOME, "temp_byexample")
output_crop_dir = os.path.join(HOME, "crop_byexample")
csv_file_path = os.path.join(HOME, "mask_labels_byexample.csv")
threshold_area = 5000
image_dir = "data_byexample_png/"
mask_dir = "mask_byexample/"
crop_dir = "crop_byexample/"
output_dir = "output_byexample/"
caption_dir = "captions_byexample/"
model_path = "runwayml/stable-diffusion-inpainting"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(caption_dir, exist_ok=True)

pbe.process_images_in_folder(data_folder, output_folder)
pbe.process_images_in_folder_(input_dir, output_mask_dir, output_temp_dir, output_crop_dir, threshold_area, csv_file_path)


if not os.path.exists(os.path.expanduser("~/.cache/huggingface/transformers")):
    print(colored(f"📝 [INFO] Downloading and caching model...", 'green'))
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
else:
    print(colored(f"📝 [INFO] Loading model from cache...", 'green'))
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

batch_size = 10
image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".png")]
total_images = len(image_filenames)

for batch_start in tqdm(range(0, total_images, batch_size), desc="Processing Batches"):
    batch_end = min(batch_start + batch_size, total_images)
    batch_filenames = image_filenames[batch_start:batch_end]

    for filename in batch_filenames:
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace('.png', 'mask0.png'))
        crop_path = os.path.join(crop_dir, filename.replace('.png', 'cropped0.png'))
        output_path = os.path.join(output_dir, filename.replace(".png", "_output.png"))

        if os.path.exists(output_path):
            print(colored(f"🛑 [INFO] Skipping {filename} as output already exists", 'yellow'))
            continue

        print(colored(f"📝 [INFO] Processing {image_path}, {mask_path}, and {crop_path}", 'green'))

        try:
            prompt, image_description, mask_description = pbg.get_image_prompt(image_path, [crop_path], des_processor, des_model)
            print(colored(f"📝 [INFO] Generated Prompt: {prompt}", 'green'))
            image = Image.open(image_path).convert("RGB")
            mask_image = Image.open(mask_path).convert("L")
            output = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
            output = output.resize(image.size)
            output.save(output_path)
            print(colored(f"✅ [INFO] Image has been saved as {output_path}", 'green'))
            caption_data = {
                "ID": os.path.splitext(os.path.basename(image_path))[0],
                "Prompt": prompt,
                "Image Description": image_description,
                "Mask description": mask_description[0]
            }
            caption_path = os.path.join(caption_dir, filename.replace(".png", ".json"))
            with open(caption_path, 'w') as caption_file:
                json.dump(caption_data, caption_file, indent=4)

            print(colored(f"✅ [INFO] Caption has been saved as {caption_path}", 'green'))
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(image)
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            axs[1].imshow(output)
            axs[1].set_title('Output Image')
            axs[1].axis('off')
            plt.show()
            image.close()
            mask_image.close()
            del image, mask_image, output
            gc.collect()
        except Exception as e:
            print(colored(f"❌ [ERROR] Failed to process {filename}: {e}", 'red'))