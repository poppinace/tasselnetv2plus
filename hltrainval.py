"""
@author: hao
"""

from utils import *
from hldataset import *
from hlnet import *
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from skimage.metrics import structural_similarity
import os
import argparse
from time import time

import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
plt.switch_backend('agg')


# prevent dataloader deadlock, uncomment if deadlock occurs
# cv.setNumThreads(0)
cudnn.enabled = True

# constant
IMG_SCALE = 1./255
IMG_MEAN = [.3405, .4747, .2418]
IMG_STD = [1, 1, 1]
SCALES = [0.7, 1, 1.3]
SHORTER_SIDE = 224

# system-related parameters
DATA_DIR = './data/maize_counting_dataset'
DATASET = 'mtc'
EXP = 'tasselnetv2plus_rf110_i64o8_r0125_crop256_lr-2_bs9_epoch500'
DATA_LIST = './data/maize_counting_dataset/train.txt'
DATA_VAL_LIST = './data/maize_counting_dataset/test.txt'

RESTORE_FROM = 'model_best.pth.tar'
SNAPSHOT_DIR = './snapshots'
RESULT_DIR = './results'

# model-related parameters
INPUT_SIZE = 64
OUTPUT_STRIDE = 8
MODEL = 'tasselnetv2plus'
RESIZE_RATIO = 0.125

# training-related parameters
OPTIMIZER = 'sgd'  # choice in ['sgd', 'adam']
BATCH_SIZE = 9
CROP_SIZE = (256, 256)
LEARNING_RATE = 1e-2
MILESTONES = [200, 400]
MOMENTUM = 0.95
MULT = 1
NUM_EPOCHS = 500
NUM_CPU_WORKERS = 0
PRINT_EVERY = 1
RANDOM_SEED = 6
WEIGHT_DECAY = 5e-4
VAL_EVERY = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# add a new entry here if creating a new data loader
dataset_list = {
    'mtc': MaizeTasselDataset,
    'wec': WhearEarDataset,
    'shc': SorghumHeadDataset,
    'uav': NewMaizeDataset
}


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Object Counting Framework")
    # constant
    parser.add_argument("--image-scale", type=float, default=IMG_SCALE,
                        help="Scale factor used in normalization.")
    parser.add_argument("--image-mean", nargs='+', type=float,
                        default=IMG_MEAN, help="Mean used in normalization.")
    parser.add_argument("--image-std", nargs='+', type=float,
                        default=IMG_STD, help="Std used in normalization.")
    parser.add_argument("--scales", type=int,
                        default=SCALES, help="Scales of crop.")
    parser.add_argument("--shorter-side", type=int,
                        default=SHORTER_SIDE, help="Shorter side of the image.")
    # system-related parameters
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str,
                        default=DATASET, help="Dataset type.")
    parser.add_argument("--exp", type=str, default=EXP,
                        help="Experiment path.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-val-list", type=str, default=DATA_VAL_LIST,
                        help="Path to the file listing the images in the val dataset.")
    parser.add_argument("--restore-from", type=str,
                        default=RESTORE_FROM, help="Name of restored model.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Where to save inferred results.")
    parser.add_argument("--save-output", action="store_true",
                        help="Whether to save the output.")
    # model-related parameters
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="the minimum input size of the model.")
    parser.add_argument("--output-stride", type=int,
                        default=OUTPUT_STRIDE, help="Output stride of the model.")
    parser.add_argument("--resize-ratio", type=float,
                        default=RESIZE_RATIO, help="Resizing ratio.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="model to be chosen.")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="Whether to use pretrained model.")
    parser.add_argument("--freeze-bn", action="store_true",
                        help="Whether to freeze encoder bnorm layers.")
    parser.add_argument("--sync-bn", action="store_true",
                        help="Whether to apply synchronized batch normalization.")
    # training-related parameters
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER,
                        choices=['sgd', 'adam'], help="Choose optimizer.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--milestones", nargs='+', type=int,
                        default=MILESTONES, help="Multistep policy.")
    parser.add_argument("--crop-size", nargs='+', type=int,
                        default=CROP_SIZE, help="Size of crop.")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Whether to perform evaluation.")
    parser.add_argument("--learning-rate", type=float,
                        default=LEARNING_RATE, help="Base learning rate for training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--mult", type=float, default=MULT,
                        help="LR multiplier for pretrained layers.")
    parser.add_argument("--num-epochs", type=int,
                        default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int,
                        default=NUM_CPU_WORKERS, help="Number of CPU cores used.")
    parser.add_argument("--print-every", type=int,
                        default=PRINT_EVERY, help="Print information every often.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--val-every", type=int, default=VAL_EVERY,
                        help="How often performing validation.")
    return parser.parse_args()


def save_checkpoint(state, snapshot_dir, filename='model_ckpt.pth.tar'):
    torch.save(state, '{}/{}'.format(snapshot_dir, filename))


def plot_learning_curves(net, dir_to_save):
    # plot learning curves
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(net.train_loss['epoch_loss'],
             label='train loss', color='tab:blue')
    ax1.legend(loc='upper right')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(net.val_loss['epoch_loss'], label='val mae', color='tab:orange')
    ax2.legend(loc='upper right')
    # ax2.set_ylim((0,50))
    fig.savefig(os.path.join(dir_to_save, 'learning_curves.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def train(net, train_loader, criterion, optimizer, epoch, args):
    # switch to 'train' mode
    net.train()

    # uncomment the following line if the training images don't have the same size
    cudnn.benchmark = True

    if args.batch_size == 1:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    running_loss = 0.0
    avg_frame_rate = 0.0
    in_sz = args.input_size
    os = args.output_stride
    target_filter = torch.cuda.FloatTensor(1, 1, in_sz, in_sz).fill_(1)
    for i, sample in enumerate(train_loader):
        torch.cuda.synchronize()
        start = time()

        inputs, targets = sample['image'], sample['target']
        inputs, targets = inputs.cuda(), targets.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs, is_normalize=False)
        # generate targets
        targets = F.conv2d(targets, target_filter, stride=os)
        # compute loss
        loss = criterion(outputs, targets)

        # backward + optimize
        loss.backward()
        optimizer.step()
        # collect and print statistics
        running_loss += loss.item()

        torch.cuda.synchronize()
        end = time()

        running_frame_rate = args.batch_size * float(1 / (end - start))
        avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
        if i % args.print_every == args.print_every-1:
            print('epoch: %d, train: %d/%d, '
                  'loss: %.5f, frame: %.2fHz/%.2fHz' % (
                      epoch,
                      i+1,
                      len(train_loader),
                      running_loss / (i+1),
                      running_frame_rate,
                      avg_frame_rate
                  ))
    net.train_loss['epoch_loss'].append(running_loss / (i+1))


def validate(net, valset, val_loader, criterion, epoch, args):
    # switch to 'eval' mode
    net.eval()
    cudnn.benchmark = False

    image_list = valset.image_list

    if args.save_output:
        epoch_result_dir = os.path.join(args.result_dir, str(epoch))
        if not os.path.exists(epoch_result_dir):
            os.makedirs(epoch_result_dir)
        cmap = plt.cm.get_cmap('jet')

    pd_counts = []
    gt_counts = []
    with torch.no_grad():
        avg_frame_rate = 0.0
        for i, sample in enumerate(val_loader):
            torch.cuda.synchronize()
            start = time()

            image, gtcount = sample['image'], sample['gtcount']
            # inference
            output = net(image.cuda(), is_normalize=not args.save_output)
            if args.save_output:
                output_save = output
                # normalization
                output = Normalizer.gpu_normalizer(output, image.size()[2], image.size()[
                                                   3], args.input_size, args.output_stride)
            # postprocessing
            output = np.clip(output, 0, None)

            pdcount = output.sum()
            gtcount = float(gtcount.numpy())

            if args.save_output:
                _, image_name = os.path.split(image_list[i])
                output_save = np.clip(
                    output_save.squeeze().cpu().numpy(), 0, None)
                output_save = recover_countmap(
                    output_save, image, args.input_size, args.output_stride)
                output_save = output_save / (output_save.max() + 1e-12)
                output_save = cmap(output_save) * 255.
                # image composition
                image = valset.images[image_list[i]]
                nh, nw = output_save.shape[:2]
                image = cv2.resize(
                    image, (nw, nh), interpolation=cv2.INTER_CUBIC)
                output_save = 0.5 * image + 0.5 * output_save[:, :, 0:3]

                dotimage = valset.dotimages[image_list[i]]

                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(dotimage.astype(np.uint8))
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(output_save.astype(np.uint8))
                ax2.get_xaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)
                fig.suptitle('manual count=%4.2f, inferred count=%4.2f' %
                             (gtcount, pdcount), fontsize=10)
                if args.dataset == 'mtc':
                    # maize tassels counting
                    plt.tight_layout(rect=[0, 0, 1, 1.4])
                elif args.dataset == 'wec':
                    # wheat ears counting
                    plt.tight_layout(rect=[0, 0, 1, 1.45])
                elif args.dataset == 'shc':
                    # sorghum heads counting -- dataset1
                    plt.tight_layout(rect=[0, 0, 0.95, 1])
                    # plt.tight_layout(rect=[0, 0, 1.2, 1]) # sorghum heads counting -- dataset2
                plt.savefig(os.path.join(epoch_result_dir, image_name.replace(
                    '.jpg', '.png')), bbox_inches='tight', dpi=300)
                plt.close()

            # compute mae and mse
            pd_counts.append(pdcount)
            gt_counts.append(gtcount)
            mae = compute_mae(pd_counts, gt_counts)
            mse = compute_mse(pd_counts, gt_counts)
            rmae, rmse = compute_relerr(pd_counts, gt_counts)

            torch.cuda.synchronize()
            end = time()

            running_frame_rate = 1 * float(1 / (end - start))
            avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
            if i % args.print_every == args.print_every - 1:
                print(
                    'epoch: {0}, test: {1}/{2}, pre: {3:.2f}, gt:{4:.2f}, me:{5:.2f}, mae: {6:.2f}, mse: {7:.2f}, rmae: {8:.2f}%, rmse: {9:.2f}%, frame: {10:.2f}Hz/{11:.2f}Hz'
                    .format(epoch, i+1, len(val_loader), pdcount, gtcount, pdcount-gtcount, mae, mse, rmae, rmse, running_frame_rate, avg_frame_rate)
                )
            start = time()
    r2 = rsquared(pd_counts, gt_counts)
    np.save(args.snapshot_dir+'/pd.npy', pd_counts)
    np.save(args.snapshot_dir+'/gt.npy', gt_counts)
    print('epoch: {0}, mae: {1:.2f}, mse: {2:.2f}, rmae: {3:.2f}%, rmse: {4:.2f}%, r2: {5:.4f}'.format(
        epoch, mae, mse, rmae, rmse, r2))
    # write to files
    with open(os.path.join(args.snapshot_dir, args.exp+'.txt'), 'a') as f:
        print(
            'epoch: {0}, mae: {1:.2f}, mse: {2:.2f}, rmae: {3:.2f}%, rmse: {4:.2f}%, r2: {5:.4f}'.format(
                epoch, mae, mse, rmae, rmse, r2),
            file=f
        )
    with open(os.path.join(args.snapshot_dir, 'counts.txt'), 'a') as f:
        for pd, gt in zip(pd_counts, gt_counts):
            print(
                '{0} {1}'.format(pd, gt),
                file=f
            )
    # save stats
    net.val_loss['epoch_loss'].append(mae)
    net.measure['mae'].append(mae)
    net.measure['mse'].append(mse)
    net.measure['rmae'].append(rmae)
    net.measure['rmse'].append(rmse)
    net.measure['r2'].append(r2)


def main():
    args = get_arguments()

    # args.evaluate_only = True
    # args.save_output = True

    args.image_mean = np.array(args.image_mean).reshape((1, 1, 3))
    args.image_std = np.array(args.image_std).reshape((1, 1, 3))

    args.crop_size = tuple(args.crop_size) if len(
        args.crop_size) > 1 else args.crop_size

    # seeding for reproducbility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # instantiate dataset
    dataset = dataset_list[args.dataset]

    args.snapshot_dir = os.path.join(
        args.snapshot_dir, args.dataset.lower(), args.exp)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    args.result_dir = os.path.join(
        args.result_dir, args.dataset.lower(), args.exp)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.restore_from = os.path.join(args.snapshot_dir, args.restore_from)

    arguments = vars(args)
    for item in arguments:
        print(item, ':\t', arguments[item])

    # instantiate network
    net = CountingModels(
        arc=args.model,
        input_size=args.input_size,
        output_stride=args.output_stride
    )

    net = nn.DataParallel(net)
    net.cuda()

    # filter parameters
    learning_params = [p[1] for p in net.named_parameters()]
    pretrained_params = []

    # define loss function and optimizer
    criterion = nn.L1Loss(reduction='mean').cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            [
                {'params': learning_params},
                {'params': pretrained_params, 'lr': args.learning_rate / args.mult},
            ],
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            [
                {'params': learning_params},
                {'params': pretrained_params, 'lr': args.learning_rate / args.mult},
            ],
            lr=args.learning_rate
        )
    else:
        raise NotImplementedError

    # restore parameters
    start_epoch = 0
    net.train_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.val_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.measure = {
        'mae': [],
        'mse': [],
        'rmae': [],
        'rmse': [],
        'r2': []
    }
    if args.restore_from is not None:
        if os.path.isfile(args.restore_from):
            checkpoint = torch.load(args.restore_from)
            net.load_state_dict(checkpoint['state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'train_loss' in checkpoint:
                net.train_loss = checkpoint['train_loss']
            if 'val_loss' in checkpoint:
                net.val_loss = checkpoint['val_loss']
            if 'measure' in checkpoint:
                net.measure['mae'] = checkpoint['measure']['mae'] if 'mae' in checkpoint['measure'] else [
                ]
                net.measure['mse'] = checkpoint['measure']['mse'] if 'mse' in checkpoint['measure'] else [
                ]
                net.measure['rmae'] = checkpoint['measure']['rmae'] if 'rmae' in checkpoint['measure'] else [
                ]
                net.measure['rmse'] = checkpoint['measure']['rmse'] if 'rmse' in checkpoint['measure'] else [
                ]
                net.measure['r2'] = checkpoint['measure']['r2'] if 'r2' in checkpoint['measure'] else [
                ]
            print("==> load checkpoint '{}' (epoch {})"
                  .format(args.restore_from, start_epoch))
        else:
            with open(os.path.join(args.snapshot_dir, args.exp+'.txt'), 'a') as f:
                for item in arguments:
                    print(item, ':\t', arguments[item], file=f)
            print("==> no checkpoint found at '{}'".format(args.restore_from))

    # define transform
    transform_train = [
        RandomCrop(args.crop_size),
        RandomFlip(),
        Normalize(
            args.image_scale,
            args.image_mean,
            args.image_std
        ),
        ToTensor(),
        ZeroPadding(args.output_stride)
    ]
    transform_val = [
        Normalize(
            args.image_scale,
            args.image_mean,
            args.image_std
        ),
        ToTensor(),
        ZeroPadding(args.output_stride)
    ]
    composed_transform_train = transforms.Compose(transform_train)
    composed_transform_val = transforms.Compose(transform_val)

    # define dataset loader
    trainset = dataset(
        data_dir=args.data_dir,
        data_list=args.data_list,
        ratio=args.resize_ratio,
        train=True,
        transform=composed_transform_train
    )
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    valset = dataset(
        data_dir=args.data_dir,
        data_list=args.data_val_list,
        ratio=args.resize_ratio,
        train=False,
        transform=composed_transform_val
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print('alchemy start...')
    if args.evaluate_only:
        validate(net, valset, val_loader, criterion, start_epoch, args)
        return

    best_mae = 1000000.0
    resume_epoch = -1 if start_epoch == 0 else start_epoch
    scheduler = MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1, last_epoch=resume_epoch)
    for epoch in range(start_epoch, args.num_epochs):
        # train
        train(net, train_loader, criterion, optimizer, epoch+1, args)
        if epoch % args.val_every == args.val_every - 1:
            # val
            validate(net, valset, val_loader, criterion, epoch+1, args)
            # save_checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                'train_loss': net.train_loss,
                'val_loss': net.val_loss,
                'measure': net.measure
            }
            save_checkpoint(state, args.snapshot_dir,
                            filename='model_ckpt.pth.tar')
            if net.measure['mae'][-1] <= best_mae:
                save_checkpoint(state, args.snapshot_dir,
                                filename='model_best.pth.tar')
                best_mae = net.measure['mae'][-1]
                best_mse = net.measure['mse'][-1]
                best_rmae = net.measure['rmae'][-1]
                best_rmse = net.measure['rmse'][-1]
                best_r2 = net.measure['r2'][-1]
            print(args.exp+' epoch {} finished!'.format(epoch+1))
            print('best mae: {0:.2f}, best mse: {1:.2f}, best_rmae: {2:.2f}, best_rmse: {3:.2f}, best_r2: {4:.4f}'
                  .format(best_mae, best_mse, best_rmae, best_rmse, best_r2))
            plot_learning_curves(net, args.snapshot_dir)
        scheduler.step()

    print('Experiments with '+args.exp+' done!')
    with open(os.path.join(args.snapshot_dir, args.exp+'.txt'), 'a') as f:
        print(
            'best mae: {0:.2f}, best mse: {1:.2f}, best_rmae: {2:.2f}, best_rmse: {3:.2f}, best_r2: {4:.4f}'
            .format(best_mae, best_mse, best_rmae, best_rmse, best_r2),
            file=f
        )
        print(
            'overall best mae: {0:.2f}, overall best mse: {1:.2f}, overall best_rmae: {2:.2f}, overall best_rmse: {3:.2f}, overall best_r2: {4:.4f}'
            .format(min(net.measure['mae']), min(net.measure['mse']), min(net.measure['rmae']), min(net.measure['rmse']), max(net.measure['r2'])),
            file=f
        )


if __name__ == "__main__":
    main()
