from essentials import *
from packages import *
import removal_init as rm
import removal_gen as rg

data_folder = "data_removal"
output_folder = "data_removal_png"
input_dir = "data_removal_png"
output_mask_dir = "mask_removal"
output_temp_dir = "temp_removal"
output_crop_dir = "crop_removal"
threshold_area = 10000
csv_file_path = "output.csv"
image_folder = "data_removal_png"
mask_folder = "mask_removal"
output_folder_ = "output_removal"

rm.process_images_in_folder(data_folder, output_folder)
rm.process_images_in_folder_(input_dir, 
                             output_mask_dir, 
                             output_temp_dir, 
                             output_crop_dir, 
                             threshold_area, 
                             csv_file_path)
rg.process_images(image_folder, mask_folder, output_folder_)