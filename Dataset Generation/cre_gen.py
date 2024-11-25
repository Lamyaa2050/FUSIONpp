from essentials import *
from packages import *

def convert_jpg_to_png(image_path, save_path):
    img = Image.open(image_path)
    img.save(save_path)
    print(f"Converted JPEG image to PNG: {save_path}")
def process_images_in_folder(data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(data_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            input_path = os.path.join(data_folder, filename)
            output_filename = filename[:-4] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            convert_jpg_to_png(input_path, output_path)

def yolov10_detection(model, image_batch):
    results = model(image_batch)
    batch_boxes = []
    batch_labels = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = [result.names[int(cls.cpu().numpy())] for cls in result.boxes.cls]
        batch_boxes.append(boxes)
        batch_labels.append(labels)
    return batch_boxes, batch_labels

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

def non_overlapping_masks(masks_with_area_label, iou_threshold=0.5):
    selected_masks_with_area_label = []
    for mask, area, label in masks_with_area_label:
        if len(selected_masks_with_area_label) == 0:
            selected_masks_with_area_label.append((mask, area, label))
        else:
            overlap = False
            for selected_mask, _, _ in selected_masks_with_area_label:
                intersection = np.logical_and(mask, selected_mask).sum()
                union = np.logical_or(mask, selected_mask).sum()
                iou = intersection / union
                if iou > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                selected_masks_with_area_label.append((mask, area, label))
        if len(selected_masks_with_area_label) == 3:
            break
    return selected_masks_with_area_label

def process_image(image_path, yolov10_model, predictor, threshold_area, output_mask_dir, output_temp_dir, output_crop_dir, csv_writer):
    image_name = os.path.basename(image_path)
    mask_output_path = os.path.join(output_mask_dir, f"{os.path.splitext(image_name)[0]}mask0.png")

    if os.path.exists(mask_output_path):
        print(f"[INFO] 🟢 Mask already exists for {image_name}, skipping.")
        return

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolov10_boxes, labels = yolov10_detection(yolov10_model, [image])
    yolov10_boxes, labels = yolov10_boxes[0], labels[0]
    predictor.set_image(image)
    all_masks_with_area_label = []
    for box, label in zip(yolov10_boxes, labels):
        input_box = np.array(box)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        for mask in masks:
            area = mask.sum()
            if area > threshold_area:
                all_masks_with_area_label.append((mask, area, label))
    all_masks_with_area_label.sort(key=lambda x: x[1], reverse=True)

    if len(all_masks_with_area_label) < 1:
        print(f"[INFO] ⚠️ No valid masks found for {image_name}, covering bounding box area instead.")
        for box, label in zip(yolov10_boxes, labels):
            binary_mask = np.zeros_like(image[:, :, 0])
            x0, y0, x1, y1 = map(int, box)
            binary_mask[y0:y1, x0:x1] = 1
            all_masks_with_area_label.append((binary_mask, (y1-y0)*(x1-x0), label))

        if len(all_masks_with_area_label) < 1:
            print(f"[INFO] ⚠️ No valid masks found for {image_name}, deleting image.")
            os.remove(image_path)
            return

    for i, (mask, area, label) in enumerate(all_masks_with_area_label[:1]):
        if area == 0:
            continue
        binary_mask = torch.from_numpy(mask).squeeze().numpy().astype(np.uint8)
        mask_name = os.path.splitext(image_name)[0] + f"mask{i}.png"
        mask_output_path = os.path.join(output_mask_dir, mask_name)
        mask_image = Image.fromarray(binary_mask * 255)
        mask_image.save(mask_output_path)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        bbox = [int(x) for x in cv2.boundingRect(largest_contour)]
        cropped_image = image.copy()
        cropped_image[binary_mask == 0] = 0
        crop_name = os.path.splitext(image_name)[0] + f"cropped{i}.png"
        crop_output_path = os.path.join(output_crop_dir, crop_name)
        cv2.imwrite(crop_output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        blended_image = image.copy()
        blended_image[binary_mask == 1] = 0
        temp_name = os.path.splitext(image_name)[0] + f"blended{i}.png"
        temp_output_path = os.path.join(output_temp_dir, temp_name)
        cv2.imwrite(temp_output_path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.show()
        print(f"[INFO] 🟢 Image: {image_name}, Mask {i}")
        print(f"[INFO] 🟢 Bounding box: {bbox}")
        print(f"[INFO] 🟢 Segmentation mask: {largest_contour.flatten().tolist()}")
        csv_writer.writerow([image_name, f"mask_{i}", label])

    del image, yolov10_boxes, labels, masks
    torch.cuda.empty_cache()
    gc.collect()

def process_images_in_folder_(input_dir, output_mask_dir, output_temp_dir, output_crop_dir, threshold_area, csv_file_path, batch_size=4):
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_temp_dir, exist_ok=True)
    os.makedirs(output_crop_dir, exist_ok=True)
    yolov10_model = YOLO('yolov10x.pt')
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image", "Mask", "Label"])
        image_filenames = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for i in range(0, len(image_filenames), batch_size):
            batch_filenames = image_filenames[i:i+batch_size]
            for image_filename in batch_filenames:
                image_path = os.path.join(input_dir, image_filename)
                process_image(image_path, yolov10_model, predictor, threshold_area, output_mask_dir, output_temp_dir, output_crop_dir, csv_writer)
            torch.cuda.empty_cache()
            gc.collect()

    input_images = set(os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png')))
    output_masks = set(os.path.splitext(f)[0] for f in os.listdir(output_mask_dir) if f.endswith('.png'))
    missing_masks = input_images - output_masks
    extra_masks = output_masks - input_images
    table_data = [
        ["Total Images", len(input_images)],
        ["Total Masks", len(output_masks)],
        ["Missing Masks", len(missing_masks)],
        ["Extra Masks", len(extra_masks)]
    ]
    print(tabulate(table_data, headers=["Description", "Count"]))
    if missing_masks:
        print(f"[INFO] 🟡 Missing masks for images: {missing_masks}")
    if extra_masks:
        print(f"[INFO] 🟡 Extra masks found: {extra_masks}")

def generate_prompt(image):
    inputs = des_processor(images=image, return_tensors="pt").to(device, torch.float32)
    out = des_model.generate(**inputs)
    image_description = des_processor.decode(out[0], skip_special_tokens=True)
    print(f"[DEBUG] 📝 Image description: {image_description}")

    chatgpt_prompt = (
        f"Given the image description: '{image_description}', suggest a specific object "
        f"that would enhance the image. The object should be easily recognizable and should "
        f"not introduce complexity to the image."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in suggesting simple, specific objects to enhance images based on descriptions."},
            {"role": "user", "content": chatgpt_prompt}
        ],
        max_tokens=50,
        temperature=0.7
    )
    suggestion = response.choices[0].message.content.strip()
    final_prompt = f"Add '{suggestion}' to the image."
    print(f"[DEBUG] 💡 Generated prompt: {final_prompt}")
    return final_prompt, image_description

def compute_difference(img1, img2):
    print("[INFO] 🔍 Computing pixel-wise difference with enhanced preprocessing...")
    time.sleep(1)
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    lab1_blur = cv2.GaussianBlur(lab1, (9, 9), 0)
    lab2_blur = cv2.GaussianBlur(lab2, (9, 9), 0)
    diff = cv2.absdiff(lab1_blur, lab2_blur)
    l, a, b = cv2.split(diff)
    gray_diff = cv2.addWeighted(l, 0.5, a, 0.25, 0)
    gray_diff = cv2.addWeighted(gray_diff, 1.0, b, 0.25, 0)
    _, threshold_mask = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("[INFO] ✅ Difference computed with adaptive thresholding.")
    return threshold_mask

def clean_mask(mask):
    print("[INFO] 🧼 Cleaning mask with advanced morphological operations...")
    time.sleep(1)
    kernel = np.ones((7, 7), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    print("[INFO] ✅ Mask cleaned.")
    return mask_cleaned

def filter_contours(mask, min_contour_area=1000):
    print("[INFO] ✂️ Filtering contours with advanced precision...")
    time.sleep(1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    print(f"[INFO] ✅ {len(contours)} contours found, filtered based on area criteria.")
    return filtered_mask

def process_pixel_diff(img1, img2):
    pixel_diff_mask = compute_difference(img1, img2)
    pixel_diff_mask_cleaned = clean_mask(pixel_diff_mask)
    final_mask = filter_contours(pixel_diff_mask_cleaned)
    return final_mask

def plot_and_save_results(original, modified, final_mask, output_path):
    print(f"[INFO] 🖼️ Plotting and saving results to {output_path}...")
    time.sleep(1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[1].imshow(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Modified Image')
    ax[2].imshow(final_mask, cmap='gray')
    ax[2].set_title('Final Mask')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()
    cv2.imwrite(output_path, final_mask)
    print("[INFO] 💾 Results saved successfully.")

def process_batch(image_batch, output_mask_dir):
    print("[INFO] 🚀 Processing batch...")
    for original_img_path, modified_img_path in tqdm(image_batch, desc="[INFO] 🖼️ Processing images in batch"):
        original_img_name = os.path.basename(original_img_path)
        output_mask_path = os.path.join(output_mask_dir, f'mask_{original_img_name}')
        if os.path.exists(output_mask_path):
            print(f"[INFO] 🟡 Mask already exists for {original_img_name}, skipping...")
            continue
        original_img = cv2.imread(original_img_path)
        modified_img = cv2.imread(modified_img_path)
        final_mask = None
        try:
            final_mask = process_pixel_diff(original_img, modified_img)
            plot_and_save_results(original_img, modified_img, final_mask, output_mask_path)
            print(f"[INFO] ✅ Mask generated for {original_img_name}.")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to process {original_img_name}: {e}")
        finally:
            del original_img, modified_img
            if final_mask is not None:
                del final_mask
            gc.collect()

def process_folder(original_img_dir, modified_img_dir, output_mask_dir, batch_size=5):
    os.makedirs(output_mask_dir, exist_ok=True)
    original_images = sorted([os.path.join(original_img_dir, f) for f in os.listdir(original_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    modified_images = sorted([os.path.join(modified_img_dir, f) for f in os.listdir(modified_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    total_batches = len(original_images) // batch_size + (1 if len(original_images) % batch_size != 0 else 0)
    print(f"[INFO] 📂 Total batches to process: {total_batches}")
    for i in tqdm(range(0, len(original_images), batch_size), desc="[INFO] 📦 Processing all batches"):
        image_batch = list(zip(original_images[i:i + batch_size], modified_images[i:i + batch_size]))
        process_batch(image_batch, output_mask_dir)
        print(f"[INFO] 🔄 Batch {i // batch_size + 1}/{total_batches} processed.")
        gc.collect()


