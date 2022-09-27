import os
import shutil
import sys
from pathlib import Path
import numpy as np
from CAT_Net.tools.infer import cat_pred


# Input Examples [NEED TO CHANGE THIS ACCCORDINGLY)]
cat_net_root =  os.path.join(os.getcwd(), "CAT_Net")
image_folder = os.path.join(os.getcwd(), "val")
path_out_folder = os.path.join(os.getcwd(), "CAT_Net", "output_pred")
##cat_net_root = 'D:\Capstone\Capstone-IS03_PT-SureBoTv2\Code\CAT_Net'
##image_folder = 'D:/Capstone/Capstone-IS03_PT-SureBoTv2/val'
##path_out_folder = 'D:\Capstone\Capstone-IS03_PT-SureBoTv2\Code\CAT-Net\output_pred'

# Catnet wrappers Function to call
def cat_inference(filename, cat_net_root):
    """
    Cat Net Wrapper for Single Image
    INPUTS:
        - path_img_input (str): input image file path 
        - path_out_folder (str): output folder for heatmap image file path, also creates a copy of original in output folder
        - path_cat_net_root (str): Root folder for CAT net, e.g. C:\CAT-Net, Can consider setting os.path.join(Path(__file__).parent, 'CAT-Net') if starting from C:\Capstone-IS03_PT-SureBoTv2\Code
    OUTPUTS:
        - pred_bool (True/False for doctoring, if True=doctored), score, heatmap
    """
    # Initialise outputs as nan 
    pred_bool = score = heatmap = np.nan
    image_folder = 'D:/Capstone/Capstone-IS03_PT-SureBoTv2/val'
    # Cat Net Folders
    print('Parsing Image to CAT-Net') 
    path_cat_net_input = os.path.join(cat_net_root, 'input')
    path_cat_net_output = os.path.join(cat_net_root, 'output_pred')
    input_image = os.path.join(image_folder, filename)
    # CAT-Net setup needs to set the working directory due to the util files [IMPT]
    os.chdir(cat_net_root)

    # Copy Img File

    if not os.path.isfile(input_image):
        print('Invalid Input Image Path')
    #file_name = os.path.split(path_img_input)
    #file_name = input_image.split("//")[-1]
    #print('file_name :', file_name)
    shutil.copy(input_image, path_cat_net_input)
    path_cat_net_input = os.path.join(path_cat_net_input, filename)
    print('path_cat_net_input :', path_cat_net_input)
    # Run Inference if valid file in input 
    if os.path.isfile(path_cat_net_input):
        print('Running CAT-Net Inference: May take several mins on CPU depending on file size')
        try:
            pred_bool, score, heatmap = cat_pred()
        except Exception as e:
            print(f'Cat Net Inference Encounter and Error: {e}')
            return
    else: 
        print('Invalid Inference Image Path')
    
    # Move Pred and Original Image to Surebot final output 
    file_name_root = os.path.splitext(filename)[0]
    path_cat_net_pred = os.path.join(path_cat_net_output, file_name_root+'.png')
    path_org_out = os.path.join(path_cat_net_output, filename)  # To handle for mix input format   
    path_pred_out = os.path.join(path_cat_net_output, file_name_root+'.png') # To handle for mix input format   
    try :
        shutil.move(path_cat_net_input, path_org_out) # Move Original Ing
    except Exception as e:
        print('Exception Error 1', e)
    try : 
        shutil.move(path_cat_net_pred, path_pred_out) # Move Predicted Img
    except Exception as e:
        print('Exception Error 2', e)
        
    print('CAT-Net Inference Completed')
    
    return pred_bool, score, heatmap

if __name__ == '__main__':

    image_name = "2002.png"
    cat_net_root = os.path.join(os.getcwd(), "CAT_Net")
    image_path = os.path.join(os.path.dirname(os.getcwd()), "images", image_name)
##    image_path = 'D:/Capstone/Capstone-IS03_PT-SureBoTv2/images/001.jpg'
##    cat_net_root = 'D:\Capstone\Capstone-IS03_PT-SureBoTv2\Code\CAT_Net'
    pred_bool, score, heatmap = cat_inference(image_path, cat_net_root)
    print(pred_bool, score, heatmap)
