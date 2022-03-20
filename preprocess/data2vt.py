import csv
from concurrent.futures import ThreadPoolExecutor

import SimpleITK as sitk
import numpy as np
import skimage.io as io
import os
import shutil
#将数据集转换成vtunet的格式,有缺失的先不做，其他直接做
from_path="F:\\Nasopharyn_Image\\Image_Data_or1\\center3"
save_path="F:\\Nasopharyn_Image\\train_data_tumor_node_240_240_155"
not_finished_list = "F:\\Nasopharyn_Image\\data_information\\not_finished_list.csv"

pool = ThreadPoolExecutor(max_workers=16)
f = open(not_finished_list, 'w',newline='',encoding='utf-8')
writer = csv.writer(f)

def read_csv(path):
    dic = dict()
    with open(path, encoding='ISO-8859-1') as t:
        f_csv = csv.DictReader(t)
        for row in f_csv:
            id = row['ID']
            metastasis = row['metastasis']
            dmfs = row['DMFS']
            dic[id] = [metastasis, dmfs]
    return dic


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data
def read3d(path):
    img = sitk.ReadImage(path)
    return img

# 显示一个系列图
def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        print(i)
        io.show()
def save_img(img, path):
    out = sitk.GetImageFromArray(img)
    sitk.WriteImage(out, path)
def list_pic_len():
    dict = {}
    for fold in os.listdir(from_path):
        dirs_path = os.path.join(from_path, fold)
        if dirs_path.endswith(""):
            os.rename(dirs_path, dirs_path.split("")[0])
            dirs_path = dirs_path.split("")[0]

        for pattern_fold in os.listdir(dirs_path):
            imgs_path = os.path.join(dirs_path, pattern_fold)
            for img in os.listdir(imgs_path):
                img_path = os.path.join(imgs_path, img)
                data = read_img(img_path)
                if dict.get(data.shape[0]) == None:
                    list = []
                    list.append(img_path)
                    dict[data.shape[0]] = list

                else:
                    dict[data.shape[0]].append(img_path)
    for d in dict.items():
        print(d)
def list_pic_size():
    dict = {}
    for fold in os.listdir(from_path):
        dirs_path = os.path.join(from_path, fold)
        if dirs_path.endswith(""):
            os.rename(dirs_path, dirs_path.split("")[0])
            dirs_path = dirs_path.split("")[0]

        for pattern_fold in os.listdir(dirs_path):
            imgs_path = os.path.join(dirs_path, pattern_fold)
            for img in os.listdir(imgs_path):
                img_path = os.path.join(imgs_path, img)
                sitkImage = sitk.ReadImage(img_path)
                size = sitkImage.GetSize()
                if dict.get(size) == None:
                    list = []
                    list.append(img_path)
                    dict[size] = list

                else:
                    dict[size].append(img_path)
    for d in dict.items():
        print(d)
def write_img(img_path, save_path):
    img = sitk.ReadImage(img_path)
    # img = resampleSize(img, 320,320,26)
    img = resampleSize(img, 240,240,155)
    sitk.WriteImage(img, save_path)
"""
统一Size
"""
def resampleSize(sitkImage, x,y,z):
    #重采样函数
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_x = xspacing/(x/float(xsize))
    new_spacing_y = yspacing/(y/float(ysize))
    new_spacing_z = zspacing/(z/float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #根据新的spacing 计算新的size
    newsize = (int(round(xsize*xspacing/new_spacing_x)),int(round(ysize*yspacing/new_spacing_y)),int(round(zsize*zspacing/new_spacing_z)))
    newspace = (new_spacing_x, new_spacing_y, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage
def multi_thread_convert_patient(path,patient, p_info):
    patient_path = os.path.join(path, patient)
    print(patient_path)
    if patient_path.endswith(""):
        os.rename(patient_path, patient_path.split("")[0])
        patient_path = patient_path.split("")[0]
    t1, t1c, t2, seg = 0, 0, 0, 0
    if patient.__contains__("不够") or patient.__contains__("补充"):
        return
    patterns = os.listdir(patient_path)
    save_patient_path = os.path.join(save_path, patient)
    if not os.path.exists(save_patient_path):
        os.makedirs(save_patient_path)



    for pattern in patterns:
        pattern_path = os.path.join(patient_path, pattern)
        if pattern.lower().__contains__("t1") and not pattern.lower().__contains__("c"):
            save_patient_t1_path = os.path.join(save_patient_path, patient + "_t1.nii.gz")
            save_seg_path = os.path.join(save_patient_path, patient + "_seg.nii.gz")
            for file in os.listdir(pattern_path):
                img_path = os.path.join(pattern_path, file)
                # t1图像保存
                if not file.lower().__contains__("label"):
                    write_img(img_path, save_patient_t1_path)
                    t1 = 1
                elif not file.lower().__contains__("node") and not file.lower().__contains__("tumo"):
                    # label图像保存
                    write_img(img_path, save_seg_path)
                    seg = 1
                elif file.lower().__contains__("tumo") and not file.lower().__contains__("node"):
                    # label的tumor图像保存
                    write_img(img_path, save_seg_path)
                    seg = 1
        if pattern.lower().__contains__("c"):
            save_patient_t1c_path = os.path.join(save_patient_path, patient + "_t1c.nii.gz")
            for file in os.listdir(pattern_path):
                # t1c图像保存
                if not file.lower().__contains__("label"):
                    img_path = os.path.join(pattern_path, file)
                    write_img(img_path, save_patient_t1c_path)
                    t1c = 1
        if pattern.lower().__contains__("t2"):
            save_patient_t2_path = os.path.join(save_patient_path, patient + "_t2.nii.gz")
            for file in os.listdir(pattern_path):
                # t2图像保存
                if not file.lower().__contains__("label"):
                    t2 = 1

                    img_path = os.path.join(pattern_path, file)
                    write_img(img_path, save_patient_t2_path)

    with open(os.path.join(save_patient_path, patient + '.txt'), 'w') as g:
        g.write(p_info[0] + " " + p_info[1])

    if t1 == 0 or t1c == 0 or t2 == 0 or seg == 0:  # 有一个文件没写，那么就报错
        writer.writerow(patient_path)
def convert2vt(path, dic):
    for patient in os.listdir(path):
        if patient in dic:
            p_info = dic[patient]
            pool.submit(multi_thread_convert_patient,path,patient,p_info)
            # multi_thread_convert_patient(path,patient,p_info)
            print(f"{patient} finshed")






# data = read_img(open_path)
# show_img(data)

if __name__ == '__main__':
    dic1 = read_csv("F:\\Nasopharyn_Image\\Image_Data_or1\\center1.csv")
    dic2 = read_csv("F:\\Nasopharyn_Image\\Image_Data_or1\\center2.csv")
    convert2vt("F:\\Nasopharyn_Image\\Image_Data_or1\\center1", dic = dic1)
    convert2vt("F:\\Nasopharyn_Image\\Image_Data_or1\\center2", dic = dic2)
    # convert2vt("F:\\Nasopharyn_Image\\Image_Data_or1\\center3")

    # from_1 = "F:\\Nasopharyn_Image\\Image_Data_or1\\center1\\ME120731MR1026\\t2\\2 OAXI T2 FSE.nii.gz"
    # to = "F:\\Nasopharyn_Image\\train_data\\ME120731MR1026\\ME120731MR1026_t2.nii.gz"
    # write_img(from_1,to)
    # list_pic_size()

    # multi_thread_convert_patient("F:\\Nasopharyn_Image\\Image_Data_or1\\center2","134892")




    pool.shutdown()
