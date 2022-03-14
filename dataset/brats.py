import os
import pathlib
import SimpleITK as sitk
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset

from config import get_brats_folder, get_test_brats_folder, get_nasopharynx_folder
from dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise


class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, on='train', data_aug=False,
                 no_seg=False, normalisation="minmax"):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.on = on
        self.datas = []
        self.validation = no_seg
        self.patterns = ["_t1", "_t1c", "_t2"]
        # self.patterns = ["_t1", "_t1ce", "_t2","_flair"]
        if not no_seg:
            self.patterns += ["_seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            metastasis, dmfs = read_information(patient_dir / f"{patient_id}.txt")

            patient = dict(
                id=patient_id, t1=paths[0], t1c=paths[1],
                t2=paths[2], seg=paths[3] if not no_seg else None,
                metastasis=int(metastasis), dmfs=int(dmfs)
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]

        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if
                         key not in ["id", "seg", "dmfs", "metastasis"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["seg"] is not None:
            """ brats的数据集处理的方式
                # et = patient_label == 4
                # et_present = 1 if np.sum(et) >= 1 else 0
                # tc = np.logical_or(patient_label == 4, patient_label == 1)
                # wt = np.logical_or(tc, patient_label == 2)
            """
            # 肿瘤标签
            tumor = patient_label == 1
            et_present = 1 if np.sum(tumor) >= 1 else 0
            # 淋巴标签，如果存在标签2，则表示淋巴标签
            node = patient_label == 2

            patient_label = np.stack([tumor, node])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.on == 'train' or self.on =='val':
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]


        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]

        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    metastasis=_patient['metastasis'],
                    dmfs=_patient['dmfs']
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)


def get_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
    base_folder = pathlib.Path(get_nasopharynx_folder(on)).resolve()
    print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])

    kfold = KFold(5, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    len_val = len(val_idx)
    val_index = val_idx[: len_val]
    # test_index = val_idx[len_val // 2 :]

    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_index]
    # test = [patients_dir[i] for i in test_index]

    # return patients_dir
    train_dataset = Brats(train, on='train',
                          normalisation=normalisation)
    val_dataset = Brats(val, on='val', data_aug=False,
                        normalisation=normalisation)
    # bench_dataset = Brats(test, training=False, benchmarking=True,
    #                       normalisation=normalisation)
    return train_dataset, val_dataset


def get_test_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
    base_folder = pathlib.Path(get_test_brats_folder()).resolve()
    print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])

    bench_dataset = Brats(patients_dir, training=False, benchmarking=True, on=on,
                          normalisation=normalisation)
    return bench_dataset


def read_information(path):
    with open(path, 'r') as f:
        sentence = f.readline()
        metastasis, dmfs = sentence.split(' ')[0], sentence.split(' ')[1]
    return metastasis, dmfs


def get_sorted_true_surival_list(datasets):
    list = []
    for data in datasets.datas:
        list.append([data['id'], data['metastasis'], data['dmfs']])
    list.sort(key=lambda a: a[2])
    return list


def write_patient_csv(datas: list, path):
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [data[0] for data in datas[0].items()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for data in datas:
            writer.writerow(data)
