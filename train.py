import argparse
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from monai.data import decollate_batch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset.brats import get_sorted_true_surival_list, get_datasets, write_patient_csv
from loss.loss import JointLoss
from loss.survival import SurvivalLoss
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch, cal_cindex
from net.vision_transformer import SWT_JSP

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
device = 'cpu'
if device == 'gpu':
    torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='VTUNET BRATS 2021 Training')
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--val', default=1, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/vt_unet_tiny.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint(0:none, 1: vt-unet model best  , 2: trained model best )')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')


def main(args):
    # setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")

    args.exp_name = "logs_base"
    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    args.checkpoint_folder = pathlib.Path(f"./runs/{args.exp_name}/model_1")

    # Create model
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = get_config(args)

    if device == 'gpu':
        model_1 = SWT_JSP(config, num_classes=args.num_classes,
                      embed_dim=yaml_cfg.get("MODEL").get("SWIN").get("EMBED_DIM"),
                      win_size=yaml_cfg.get("MODEL").get("SWIN").get("WINDOW_SIZE")).cuda()
    else:
        model_1 = SWT_JSP(config, num_classes=args.num_classes,
                      embed_dim=yaml_cfg.get("MODEL").get("SWIN").get("EMBED_DIM"),
                      win_size=yaml_cfg.get("MODEL").get("SWIN").get("WINDOW_SIZE"))


    model_1.load_from(config)

    if args.resume == 2:
        args.checkpoint = args.checkpoint_folder / "model_best.pth.tar"
        reload_ckpt_bis(args.checkpoint, model_1)
    elif args.resume == 1:
        args.checkpoint = "pretrained_ckpt/vtunet_model_best.pth.tar"
        reload_ckpt_bis(args.checkpoint, model_1)


    print(f"total number of trainable parameters {count_parameters(model_1)}")
    if device == 'gpu':
        model_1 = model_1.cuda()
    else:
        model_1 = model_1

    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    if device == 'gpu':
        criterion = SurvivalLoss().cuda()
        criterian_val = SurvivalLoss().cuda()
    else:
        criterion = SurvivalLoss()
        criterian_val = SurvivalLoss()

    params = model_1.parameters()

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    full_train_dataset, l_val_dataset = get_datasets(args.seed, fold_number=args.fold)
    #     full_train_dataset, l_val_dataset, bench_dataset = get_datasets(args.seed, fold_number=args.fold)
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    # bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)

    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))

    # 获取full_train_dataset的survival序列
    true_train_survival = get_sorted_true_surival_list(full_train_dataset)

    # 获取val_dataset的survival序列
    true_val_survival = get_sorted_true_surival_list(l_val_dataset)

    # event sorted by true dmfs.
    sorted_train_event = torch.as_tensor([a[1] for a in true_train_survival])
    sorted_val_event = torch.as_tensor([a[1] for a in true_val_survival])



    #write patients information to csv.
    write_patient_csv(full_train_dataset.datas, 'patient_information/train.csv')
    write_patient_csv(l_val_dataset.datas, 'patient_information/val.csv')

    # print("Bench Test dataset number of batch:", len(bench_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_1 = 0.0
    patients_perf = []

    # Initiate risk_sequence.
    risk_sequence = torch.linspace(1, 0, len(true_train_survival))
    if device == 'gpu':
        risk_sequence = risk_sequence.cuda()

    print("start training now!")


    for epoch in range(args.epochs):
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')

            batch_per_epoch = len(train_loader)
            progress = ProgressMeter(
                batch_per_epoch,
                [batch_time, data_time, losses_,],
                prefix=f"train Epoch: [{epoch}]")


            if device == 'gpu':
                sorted_train_event = sorted_train_event.cuda()
                sorted_val_event = sorted_val_event.cuda()


            end = time.perf_counter()
            metrics = []

            for i, batch in enumerate(zip(train_loader)):

                torch.cuda.empty_cache()
                # measure data loading time
                data_time.update(time.perf_counter() - end)

                inputs_S1, seg_S1 = batch[0]["image"].float(), batch[0]["label"].float()

                inputs_S1, risk_sequence = Variable(inputs_S1), Variable(risk_sequence)
                if device == 'gpu':
                    inputs_S1, risk_sequence = inputs_S1.cuda(), risk_sequence.cuda()

                optimizer.zero_grad()

                risk = model_1(inputs_S1, seg_S1)

                #pack risk
                for k, id in enumerate(batch[0]["patient_id"]):
                    for j, t in enumerate(true_train_survival):
                        if id == t[0]:
                            risk_sequence[j] = risk[k]
                            break



                loss_ = criterion(risk_sequence, sorted_train_event)

                t_writer_1.add_scalar(f"Loss/train{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                # compute gradient and do SGD step
                loss_.mean().backward()
                optimizer.step()
                # t_writer_1.add_graph(model_1, inputs_S1)

                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)
                if scheduler is not None:
                    scheduler.step()

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                # Display progress
                progress.display(i)

            t_writer_1.add_scalar(f"SummaryLoss/train/joint", losses_.avg, epoch)

            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")
            torch.cuda.empty_cache()

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0:
                validation_loss_1, validation_cindex = step(val_loader, model_1, criterian_val, epoch, t_writer_1,
                                                          true_val_survival, sorted_val_event,
                                                          save_folder=args.save_folder_1,
                                                          patients_perf=patients_perf)

                t_writer_1.add_scalar(f"SummaryCIndex{''}", validation_cindex, epoch)
                t_writer_1.add_scalar(f"SummaryLoss", validation_loss_1, epoch)

                if validation_cindex > best_1:

                    print(f"Saving the model with DSC {validation_cindex}")
                    best_1 = validation_cindex
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ), is_best=True,
                        save_folder=args.save_folder_1, )

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")


def step(data_loader, model, criterion, epoch, writer, true_val_survival, sorted_val_event,  save_folder=None, patients_perf=None):
    risk_sequence = torch.linspace(1, 0, len(true_val_survival))
    if device == 'gpu':
        risk_sequence = risk_sequence.cuda()

    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"val Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []

    for i, val_data in enumerate(data_loader):


        # measure data loading time
        data_time.update(time.perf_counter() - end)

        patient_id = val_data["patient_id"]

        model.eval()
        with torch.no_grad():
            if device == 'gpu':
                val_inputs, val_segs = (
                    val_data["image"].cuda(),
                    val_data["label"].cuda(),
                )
            else:
                val_inputs, val_segs = (
                    val_data["image"],
                    val_data["label"],
                )
            if device == 'cpu':
                val_inputs = val_inputs.float()

            val_risk = model(val_inputs, val_segs)

            # pack risk
            for j, t in enumerate(true_val_survival):
                if patient_id == t[0]:
                    risk_sequence[j] = val_risk

            loss_ = criterion(risk_sequence, sorted_val_event)

        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split='val', loss=loss_.item())
            )

        writer.add_scalar(f"Loss/val{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)


        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")
        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    cindex = cal_cindex(true_val_survival, risk_sequence)

    writer.add_scalar(f"SummaryLoss/val/loss", losses.avg, epoch)

    print("val Epoch"+"["+str(epoch)+"]"+" c-index:"+str(cindex)+"!")

    return losses.avg, cindex


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
