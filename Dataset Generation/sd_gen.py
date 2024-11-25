from essentials import *
from packages import *

def generate_prompt(image_description, mask_description):
    user_message = f"Replace the masked area in the image where {mask_description}. The image is {image_description}."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Diffusion Models Expert in creating prompts for image editing with DALL-E."},
            {"role": "user", "content": user_message}
        ]
    )
    return completion.choices[0].message["content"]

def get_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = des_processor(images=image, return_tensors="pt")
    out = des_model.generate(**inputs)
    return des_processor.decode(out[0], skip_special_tokens=True)

def image_to_tensor(image, mask=False):
    np_image = np.array(image)
    if mask:
        if np_image.ndim == 2:
            np_image = np.expand_dims(np_image, axis=-1)
        return torch.tensor(np_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    else:
        return torch.tensor(np_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

def process_images_in_batches(image_paths, batch_size=5):
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        for image_path in batch:
            try:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                mask_path = os.path.join(mask_dir, f"{image_name}_mask.png")
                cropped_path = os.path.join(crop_dir, f"{image_name}_cropped.png")

                if not os.path.exists(mask_path) or not os.path.exists(cropped_path):
                    continue

                image = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                image_description = get_description(image_path)
                mask_description = get_description(cropped_path)

                print(f"[INFO] 🖼️ Image Description: {image_description}")
                print(f"[INFO] 🎭 Mask Description: {mask_description}")

                prompt = generate_prompt(image_description, mask_description)

                print(f"[INFO] 📝 Generated Prompt: {prompt}")

                image_tensor = image_to_tensor(image)
                mask_tensor = image_to_tensor(mask, mask=True)


                if not isinstance(image_tensor, torch.Tensor):
                    raise TypeError(f"Expected image_tensor to be torch.Tensor but got {type(image_tensor)}")
                if not isinstance(mask_tensor, torch.Tensor):
                    raise TypeError(f"Expected mask_tensor to be torch.Tensor but got {type(mask_tensor)}")

                inpainted_image = sd_pipeline(prompt=prompt, image=image_tensor, mask=mask_tensor).images[0]
                inpainted_image.save(os.path.join(output_dir, f"{image_name}_inpainted.png"))

                caption_data = {
                    "ID": image_name,
                    "Prompt": prompt,
                    "Image Description": image_description,
                    "Mask description": mask_description
                }
                with open(os.path.join(caption_dir, f"{image_name}.json"), 'w') as caption_file:
                    json.dump(caption_data, caption_file)

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(image)
                axs[0].set_title('Original Image')
                axs[0].axis('off')
                axs[1].imshow(inpainted_image)
                axs[1].set_title('Inpainted Image')
                axs[1].axis('off')
                plt.show()

            except Exception as e:
                print(f"[ERROR] ❌ Error processing {image_path}: {e}")

        gc.collect()

