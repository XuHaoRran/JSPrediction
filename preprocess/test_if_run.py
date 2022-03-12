import os
import pathlib
import torch
from dataset import Brats
from torch.autograd import Variable
from dataset.brats import get_datasets

def get_my_datasets(normalisation="minmax"):
    path = "F:\\Nasopharyn_Image\\train"
    base_folder = pathlib.Path(path).resolve()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    train = []
    print(patients_dir)
    for dir in patients_dir:
        train.append(dir)

    # return patients_dir
    train_dataset = Brats(train, training=False,
                          normalisation=normalisation)
    return train_dataset
if __name__ == '__main__':
    val_dataset = get_my_datasets()
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)

    # full_train_dataset, l_val_dataset, bench_dataset = get_datasets(1234, fold_number=0)
    # train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=2, shuffle=True,
    #                                            num_workers=2, pin_memory=True, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=2, shuffle=False,
    #                                          pin_memory=True, num_workers=1)
    # bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=2, num_workers=2)
    # print("Train dataset number of batch:", len(train_loader))
    # print("val_loader dataset number of batch:", len(val_loader))
    # print("bench_loader dataset number of batch:", len(bench_loader))
    # print(l_val_dataset.datas)
    for i, batch in enumerate(zip(val_loader)):
        print(i)
        print(batch[0]["patient_id"])

        inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["label"].float()
        inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)

    # for i, batch in enumerate(zip(test)):
    #     print(i)
    #     inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["label"].float()
    #     inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
    print("asdfsadfadf")
