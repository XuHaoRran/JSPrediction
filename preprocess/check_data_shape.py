import os
import time
from concurrent.futures import ThreadPoolExecutor

import SimpleITK as sitk
import numpy as np
import skimage.io as io
import os
import shutil
import csv
train_data_path="F:\\Nasopharyn_Image\\train_data"

z = (240,240,155)
pool = ThreadPoolExecutor(max_workers=16)

def get_one_shape(patient_path):
    list = []
    for file in os.listdir(patient_path):
        img_path = os.path.join(patient_path, file)
        sitkImage = sitk.ReadImage(img_path)
        size = sitkImage.GetSize()
        list.append(size)
        if z != size:
            print(img_path,size)
        print(img_path,size)

def get_one_label(patient_path):
    list = []
    for file in os.listdir(patient_path):
        if not file.__contains__("seg"):
            continue
        img_path = os.path.join(patient_path, file)
        sitkImage = sitk.ReadImage(img_path)
        label = sitk.GetArrayFromImage(sitkImage)
        unique, counts = np.unique(label, return_counts=True)
        dic = dict(zip(unique, counts))
        # assert len(dic == 2)
        print(dic[1])
        if (dic[0] <= dic[1]):
            print(img_path)

def get_all_data(path):
    for patient in os.listdir(path):
        patient_path = os.path.join(path, patient)
        pool.submit(get_one_shape,patient_path)



if __name__ == '__main__':
    get_all_data(train_data_path)

    # img_path = "F:\\Nasopharyn_Image\\train_data\\ME100609MR1031\\ME100609MR1031_seg.nii.gz"
    # sitkImage = sitk.ReadImage(img_path)
    # size = sitkImage.GetSize()
    # print(size)




