from essentials import *
from packages import *

def yolov10_detection(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image, stream=True)
    for result in results:
        boxes = result.boxes
        labels = result.names
    return boxes.xyxy.tolist(), labels, image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def process_image(image_path, yolov10_model, predictor, output_mask_dir, output_temp_dir, output_crop_dir):
    yolov10_boxes, label_indices, image = yolov10_detection(yolov10_model, image_path)
    predictor.set_image(image)
    labels = yolov10_model.names
    for box, label_index in zip(yolov10_boxes, label_indices):
        label_name = labels[label_index]
        input_box = np.array(box)
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=True)
        masks_with_areas = [(mask, np.sum(mask)) for mask in masks]
        largest_mask, _ = max(masks_with_areas, key=lambda x: x[1])
        binary_mask = torch.from_numpy(largest_mask).squeeze().numpy().astype(np.uint8)
        mask_image = Image.fromarray(binary_mask * 255)
        image_name = os.path.basename(image_path)
        mask_name = os.path.splitext(image_name)[0] + f"_mask.png"
        mask_output_path = os.path.join(output_mask_dir, mask_name)
        mask_image.save(mask_output_path)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Warning: No contours found for the largest mask in image {image_name}. Skipping...")
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        bbox = [int(x) for x in cv2.boundingRect(largest_contour)]
        cropped_image = image.copy()
        cropped_image[binary_mask == 0] = 0
        crop_name = os.path.splitext(image_name)[0] + f"_cropped.png"
        crop_output_path = os.path.join(output_crop_dir, crop_name)
        cv2.imwrite(crop_output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        blended_image = image.copy()
        blended_image[binary_mask == 1] = 0
        temp_name = os.path.splitext(image_name)[0] + f"_blended.png"
        temp_output_path = os.path.join(output_temp_dir, temp_name)
        cv2.imwrite(temp_output_path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(largest_mask, plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.show()
        print(f"Image: {image_name}")
        print("Bounding box:", bbox)
        print("Segmentation mask:", largest_contour.flatten().tolist())

def process_images_in_folder(input_dir, output_mask_dir, output_temp_dir, output_crop_dir):
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_temp_dir, exist_ok=True)
    os.makedirs(output_crop_dir, exist_ok=True)
    yolov10_model = YOLO('yolov10x.pt')
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    for image_filename in os.listdir(input_dir):
        if image_filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, image_filename)
            process_image(image_path, yolov10_model, predictor, output_mask_dir, output_temp_dir, output_crop_dir)

def get_image_prompt(image_path, mask_paths, crop_paths, processor, model, actions):
    image = Image.open(image_path).convert('RGB')
    descriptions = []
    for crop_path, action in zip(crop_paths, actions):
        crop = Image.open(crop_path).convert('RGB')
        cropped_image_inputs = processor(images=crop, return_tensors="pt")
        cropped_image_out = model.generate(**cropped_image_inputs)
        cropped_object_description = processor.batch_decode(cropped_image_out, skip_special_tokens=True)[0]
        descriptions.append(cropped_object_description)
    image_inputs = processor(images=image, return_tensors="pt")
    image_out = model.generate(**image_inputs)
    image_description = processor.batch_decode(image_out, skip_special_tokens=True)[0]
    print(f"Image Description: {image_description}")
    print(f"Mask Description: {cropped_object_description}")
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
    return completion.choices[0].message.content.strip()

def inpaint_image(image_path, crop_paths, actions, steps=100, guidance_scale=7.5):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    aspect_ratio = width / height
    new_width = width - width % 8
    new_height = int(new_width / aspect_ratio) - int(new_width / aspect_ratio) % 8
    remove_prompt = get_image_prompt(image_path, crop_paths, des_processor, des_model, actions)
    print(f"Generated Prompt: {remove_prompt}")
    promptA = f"P_ctxt {remove_prompt}"
    promptB = f"P_ctxt {remove_prompt}"
    negative_promptA = f"P_obj"
    negative_promptB = f"P_obj"
    crops_resized = [Image.open(crop_path).convert("RGB").resize((new_width, new_height)) for crop_path in crop_paths]

    inpainted_image = pipe(
        promptA=promptA,
        promptB=promptB,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=image.resize((new_width, new_height)),
        imageA=crops_resized[0],
        imageB=crops_resized[0],
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        tradoff=0.0,
        tradoff_nag=1.0
    ).images[0]
    for crop in crops_resized[1:]:
        inpainted_image = pipe(
            promptA=promptA,
            promptB=promptB,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            image=inpainted_image,
            imageA=crop,
            imageB=crop,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            tradoff=0.0,
            tradoff_nag=1.0
        ).images[0]

    inpainted_image = inpainted_image.resize((width, height), resample=Image.LANCZOS)
    return inpainted_image

final_folder = "final"
def display_pp_manipulations(data_folder, output_folder, levels=["replacement", "removement", "comb3", "comb4"]):
    data_images = sorted([f for f in os.listdir(data_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    num_levels = len(levels)
    for data_img in data_images:
        data_path = os.path.join(data_folder, data_img)
        original_image = Image.open(data_path).convert("RGB")
        rows = (num_levels + 1) // 5 + 1
        cols = min(num_levels + 1, 5)
        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image)
        ax1.set_title('Original Image', fontsize=10)
        ax1.axis('off')
        for i, level in enumerate(levels):
            output_img = data_img.replace(".png", f"_{level}.png")
            output_path = os.path.join(output_folder, output_img)
            try:
                generated_image = Image.open(output_path).convert("RGB")
            except FileNotFoundError:
                print(f"Warning: {level} image '{output_img}' not found for '{data_img}'.")
                continue
            row = (i + 1) // 5
            col = (i + 1) % 5
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(generated_image)
            ax.set_title(f"DALLE - {level.capitalize()}", fontsize=10)
            ax.axis('off')
        tit = data_img.split(".png")[0]
        levels_str = "_".join([level.upper() for level in levels])
        output_plot_path = os.path.join(final_folder, f"{data_img[:-4]}_{levels_str}.png")
        fig.suptitle(f"PowerPaint Manipulations for {tit}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        print(f"Plot saved to: {output_plot_path}")
        plt.show()

