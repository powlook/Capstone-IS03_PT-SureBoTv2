import os, warnings, time
import shutil
import sys
from glob import glob
from pathlib import Path
from pyfiglet import Figlet
import numpy as np
import pandas as pd
from CAT_Net.tools.infer import cat_pred
warnings.filterwarnings("ignore")

# Catnet wrappers Function to call
def cat_inference(filename, cat_net_root):
    """
    Cat Net Wrapper for Single Image
    INPUTS:
        - image_folder (str): input image file path 
        - path_cat_net_output (str): output folder for heatmap image file path, also creates a copy of original in output folder
        - cat_net_root (str): Root folder for CAT net, e.g. C:\CAT-Net, Can consider setting os.path.join(Path(__file__).parent, 'CAT-Net') if starting from C:\Capstone-IS03_PT-SureBoTv2\Code
    OUTPUTS:
        - pred_bool (True/False for doctoring, if True=doctored), score, heatmap
        - score
    """
    # Initialise outputs as nan 
    pred_bool = score = heatmap = np.nan
    # Cat Net Folders
    print('Parsing Image to CAT-Net')
    image_folder = os.path.join(os.getcwd(), "images")
    path_cat_net_input = os.path.join(cat_net_root, 'input')
    path_cat_net_output = os.path.join(cat_net_root, 'output_pred')
    input_image = os.path.join(image_folder, filename)
    # CAT-Net setup needs to set the working directory due to the util files [IMPT]
    os.chdir(cat_net_root)

    # Copy Img File

    if not os.path.isfile(input_image):
        print('Invalid Input Image Path')
    shutil.copy(input_image, path_cat_net_input)
    path_cat_net_input = os.path.join(path_cat_net_input, filename)
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
    file_name_root = filename.split('\\')[-1].split('.')[0]
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


def main():
    
    custom_fig = Figlet(font='standard')
    catnet_banner = custom_fig.renderText("CAT_Net")
    print('\n')
    print(catnet_banner)
    picture_folder = '../images'
    picture_list = os.listdir(picture_folder)
    while True:
        os.chdir('D:\Capstone\Capstone-IS03_PT-SureBoTv2\Code')
        print("\n===== NEW QUERY =====")
        file_name = str(input("Filename: "))
        if file_name in picture_list:
            #img_filepath = os.path.join(picture_folder, file_name)
            #img = cv2.imread(img_filepath, cv2.IMREAD_ANYCOLOR)
            #cv2.imshow('Picture', img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            cat_net_root = os.path.join(os.getcwd(), "CAT_Net")
            image_path = os.path.join(os.path.dirname(os.getcwd()), "images", file_name)
            pred_bool, score, heatmap = cat_inference(image_path, cat_net_root)
            print('Predicted Outcome : ', pred_bool)
            print('Score             :', score)
        else:
            print('Filename not in images folder')


def main_2():
    
    cat_net_root = os.path.join(os.getcwd(), "CAT_Net")
    picture_folder = os.getcwd()+'/../val/*.*'
    print('picture_folder :', picture_folder)
    files = glob(picture_folder)
    df = pd.DataFrame(columns=['filename', 'cat_net', 'score','ground_truth', 'time_taken(mins)'])
    
    for i, img_filepath in enumerate(files[0:5]):
        #img_filepath = os.path.join(picture_folder, file)
        s_time = time.time()
        file = img_filepath.split('\\')[-1]
        print(f'\n***** Infer New file file_name : {file} *****')
        if file[0] == '0':
            ground_truth = 'SUPPORTS'
        elif file[0] == '2':
            ground_truth = 'REFUTES'
        else: ground_truth = 'NONE'

        pred_bool, score, heatmap = cat_inference(img_filepath, cat_net_root)
        
        time_taken = round((time.time() - s_time)/60,2)
        print('Results :', i, file, pred_bool, score, ground_truth, time_taken)
        
        df.loc[i] = file, pred_bool, score, ground_truth, time_taken

    df.to_excel('validate_cat_net_5_10.xlsx')
    
if __name__ == '__main__':

    print('Are you infering individual images or groups of images in a folder')
    ans = int(input("[1] one image, [2] folder of images :"))
    print(ans)
    if ans == 1:   
        main()
    elif ans == 2:
        main_2()
    else:
        print('Invalid Input. Try again')