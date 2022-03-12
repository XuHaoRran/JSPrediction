import os
import SimpleITK as sitk
import numpy as np
import skimage.io as io
import os
import shutil
import csv
center1_path = "F:\\Nasopharyn_Image\\Image_Data_or1\\center1"
center2_path = "F:\\Nasopharyn_Image\\Image_Data_or1\\center2"
save_path = "F:\\Nasopharyn_Image\\data_information\\data_list.csv"


f = open(save_path, 'w',newline='',encoding='utf-8')
writer = csv.writer(f)
#将文件信息写入文件，这个信息用来对照哪个哪个文件还没被处理
def get_data_information_list(path):
    seg_img = ""
    t1_img = ""
    t1pc_img = ""
    t2_img = ""
    for patient in os.listdir(path):

        if patient.__contains__("不够") or patient.__contains__("补充"):
            continue
        dict = {}
        dict['t1'], dict['t1c'], dict['t2'], dict['label'], dict['tumo'] = "*", "*", "*", "*", "*"
        patient_path = os.path.join(path, patient)
        for pattern in os.listdir(patient_path):
            pattern_path = os.path.join(patient_path, pattern)
            if pattern.lower().__contains__("t1") and not pattern.lower().__contains__("c"):
                for file in os.listdir(pattern_path):
                    img = os.path.join(pattern_path, file)
                    # t1图像保存
                    if not file.lower().__contains__("label"):
                        dict['t1'] = img

                    elif file.lower().__contains__("tumo") and not file.lower().__contains__("node"):

                        # label的tumor图像保存
                        dict['tumo'] = img

            if pattern.lower().__contains__("c"):
                for file in os.listdir(pattern_path):
                    if not file.lower().__contains__("label"):
                        img_path = os.path.join(pattern_path, file)
                        dict['t1c'] = img_path

            if pattern.lower().__contains__("t2"):
                for file in os.listdir(pattern_path):
                    # t1c图像保存
                    if not file.lower().__contains__("label"):
                        img_path = os.path.join(pattern_path, file)
                        dict['t2'] = img_path
        row = [patient, dict['t1'], dict['t1c'], dict['t2'], dict['tumo']]
        writer.writerow(row)

if __name__ == '__main__':
    get_data_information_list(center1_path)
    get_data_information_list(center2_path)
    # aaa = "F:\Nasopharyn_Image\train_data\ME100609MR1031\ME100609MR1031_seg.nii.gz"

    # img_path = "F:\\Nasopharyn_Image\\train_data\\ME100609MR1031\\ME100609MR1031_seg.nii.gz"
    # sitkImage = sitk.ReadImage(img_path)
    # size = sitkImage.GetSize()
    # print(size)

    f.close()



