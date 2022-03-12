import pathlib
import SimpleITK as sitk
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt

path4_folder = "F:\\Nasopharyn_Image\\train_data\\129369\\129369_seg.nii.gz"
img_folder = "F:\\Nasopharyn_Image\\train_data\\129369\\129369_t1.nii.gz"
patient4_label = sitk.GetArrayFromImage(sitk.ReadImage(str(path4_folder)))
img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_folder)))
# print(img)
# img[patient4_label == 2] = 255
# print(img[img == 255])
# np.unique(patient4_label)
# print(patient4_label.shape)
# print(patient4_label.shape)

def show_img(img):
    # img = cv2.equalizeHist(img)
    image = np.asanyarray(img, dtype=np.uint8)
    cv2.imshow('1',image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()   #cv2.destroyWindow(wname)


show_img(patient4_label[12])
