from dalle_main import *
from dalle_init import *
import dalle_init
from dalle_mask import *
import dalle_mask
from packages import *
from essentials import *

data_folder = "data_dalle"
output_folder = "data_dalle_png"
dalle_init.process_images_in_folder(data_folder, output_folder)

sys.path.append("..")
HOME = os.getcwd()
input_dir = os.path.join(HOME, "data_dalle_png")
output_mask_dir = os.path.join(HOME, "mask_dalle")
output_temp_dir = os.path.join(HOME, "temp_dalle")
output_crop_dir = os.path.join(HOME, "crop_dalle")
csv_file_path = os.path.join(HOME, "mask_labels.csv")
threshold_area = 5000
dalle_mask.process_images_in_folder(input_dir, 
                                    output_mask_dir, 
                                    output_temp_dir, 
                                    output_crop_dir, 
                                    threshold_area, 
                                    csv_file_path)

data_png_folder = "data_dalle_png"
transparent_folder = "transparent"
crop_folder = "crop_dalle"
output_dalle_creation_folder = "output_dalle"
LEVELS = ["replacement"]
process_folders(LEVELS, 
                data_png_folder, 
                transparent_folder, 
                crop_folder, 
                output_dalle_creation_folder, 
                des_processor, 
                des_model)

