from essentials import *
from packages import *

def get_image_prompt(image_path, crop_paths, processor, model, actions):
    image = Image.open(image_path).convert('RGB')
    descriptions = []
    print(f"[DEBUG] Image Path: {image_path}")
    print(f"[DEBUG] Crop Paths: {crop_paths}")
    print(f"[DEBUG] Actions: {actions}")
    for crop_path, action in zip(crop_paths, actions):
        try:
            if os.path.exists(crop_path):
                crop = Image.open(crop_path).convert('RGB')
                print(f"[INFO] 📸 Processing crop image: {crop_path}")
                masked_image_inputs = processor(images=crop, return_tensors="pt")
                masked_image_out = model.generate(**masked_image_inputs)
                masked_object_description = processor.batch_decode(masked_image_out, skip_special_tokens=True)[0]
                descriptions.append(masked_object_description)
                print(f"[INFO] 🖼️ Crop Description: {masked_object_description}")
            else:
                print(f"[WARNING] ⚠️ Crop path {crop_path} does not exist.")
                descriptions.append("")
        except Exception as e:
            print(f"[ERROR] ❌ Error processing crop path {crop_path}: {e}")
            descriptions.append("")
    image_inputs = processor(images=image, return_tensors="pt")
    image_out = model.generate(**image_inputs)
    image_description = processor.batch_decode(image_out, skip_special_tokens=True)[0]
    print(f"[INFO] 📸 Image Description: {image_description}")
    print(f"[INFO] 🖼️ Mask Descriptions (Crop Paths): {descriptions}")
    user_message_parts = []
    for desc, action in zip(descriptions, actions):
        if action == "replacement":
            user_message_parts.append(f"replace the {desc}")
        elif action == "removement":
            user_message_parts.append(f"remove the {desc}")
        elif action == "creation":
            user_message_parts.append(f"create a new object in the area previously occupied by '{desc}'")
        else:
            raise ValueError("Invalid action. Choose 'replacement', 'removal', or 'creation'.")
    user_message = f"Generate a prompt to {' and '.join(user_message_parts)} in an image similar to '{image_description}', focusing on the areas defined by the provided masks. Ensure the objects fit seamlessly into the scene."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are skilled in creating prompts for DALL-E 2 image editing."},
            {"role": "user", "content": user_message}
        ]
    )
    return completion.choices[0].message.content.strip(), image_description, descriptions

def process_image_pair(image_path, mask_paths, crop_paths, save_path, processor, model, actions, mask_rem_rep_folder, data_rem_rep_png_folder):
    try:
        if os.path.exists(save_path):
            print(f"[INFO] 🔍 Output image already exists at {save_path}. Skipping...")
            return

        prompt, image_description, descriptions = get_image_prompt(image_path, crop_paths, processor, model, actions)
        print(f"[INFO] 📝 Generated Prompt: {prompt}")
        response = openai.Image.create_edit(
            image=open(image_path, "rb"),
            mask=open(mask_paths[0], "rb"),
            prompt=prompt,
            n=1,
            size="1024x1024",
            model="dall-e-2"
        )
        image_url = response["data"][0]["url"]
        response = requests.get(image_url)
        with Image.open(BytesIO(response.content)) as img:
            original_size = Image.open(image_path).size
            img = img.resize(original_size, Image.LANCZOS)
            img.save(save_path)
            print(f"[INFO] 💾 Edited image saved to {save_path}")
            captions_folder = "captions_dalle"
            os.makedirs(captions_folder, exist_ok=True)
            json_path = os.path.join(captions_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
            caption_data = {
                "ID": os.path.splitext(os.path.basename(image_path))[0],
                "Prompt": prompt,
                "Image Description": image_description,
                "Mask Description": descriptions
            }
            with open(json_path, 'w') as json_file:
                json.dump(caption_data, json_file)
            print(f"[INFO] 📝 Captions saved to {json_path}")
            fig, axes = plt.subplots(1, 3, figsize=(20, 15))

            original_image_path = os.path.join(data_rem_rep_png_folder, os.path.basename(image_path))
            original_image = Image.open(original_image_path).convert('RGB')
            edited_image = Image.open(save_path).convert('RGB')

            mask_image_name = os.path.splitext(os.path.basename(image_path))[0] + "mask0.png"
            mask_image_path = os.path.join(mask_rem_rep_folder, mask_image_name)
            mask_image = Image.open(mask_image_path).convert('RGB')

            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(edited_image)
            axes[1].set_title("Edited Image")
            axes[1].axis('off')

            axes[2].imshow(mask_image)
            axes[2].set_title("Mask Image")
            axes[2].axis('off')

            plt.show()
    except Exception as e:
        print(f"[ERROR] ❌ Error processing {image_path}: {e}")

def process_folders_(levels, data_folder, mask_folder, crop_folder, output_folder, mask_rem_rep_folder, data_rem_rep_png_folder, processor, model):
    os.makedirs(output_folder, exist_ok=True)
    actions_dict = {
        "replacement": ["replacement"],
        "removement": ["removement"],
        "creation": ["creation"],
        "comb1": ["removement", "replacement"],
        "comb2": ["removement", "creation"],
        "comb3": ["replacement", "creation"],
        "comb4": ["removement", "replacement", "creation"]
    }

    for filename in os.listdir(data_folder):
        if filename.endswith(".png"):
            base_image_name = os.path.splitext(filename)[0]
            base_image_path = os.path.join(data_folder, filename)
            all_mask_paths = [
                os.path.join(mask_folder, f"{base_image_name}blended{i}_transparent.png")
                for i in range(3)
            ]
            all_crop_paths = [
                os.path.join(crop_folder, f"{base_image_name}cropped{i}.png")
                for i in range(3)
            ]
            print(f"[DEBUG] All Mask Paths: {all_mask_paths}")
            print(f"[DEBUG] All Crop Paths: {all_crop_paths}")
            existing_mask_paths = [mask_path for mask_path in all_mask_paths if os.path.exists(mask_path)]
            existing_crop_paths = [crop_path for crop_path in all_crop_paths if os.path.exists(crop_path)]
            print(f"[DEBUG] Existing Mask Paths: {existing_mask_paths}")
            print(f"[DEBUG] Existing Crop Paths: {existing_crop_paths}")
            for level in levels:
                actions = actions_dict[level]
                print(f"\n[INFO] 🔧 Processing action '{level}' for image '{filename}'\n" + "-" * 50)
                try:
                    if len(existing_mask_paths) >= len(actions):
                        output_path = os.path.join(output_folder, f"{base_image_name}_{level}.png")
                        process_image_pair(base_image_path, existing_mask_paths[:len(actions)], existing_crop_paths[:len(actions)], output_path, processor, model, actions, mask_rem_rep_folder, data_rem_rep_png_folder)
                    else:
                        print(f"[FAILURE] ⚠️ Required masks for {filename} and action {level} are missing.")
                except Exception as e:
                    print(f"[ERROR] ❌ Error processing {filename}: {e}")
                print("\n" + "=" * 50 + "\n")

