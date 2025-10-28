# config.py
import argparse
import os
import torch


def get_args():
    parser = argparse.ArgumentParser()

    """PACS"""
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--source-domain', nargs='+', default=['photo', 'cartoon', 'art_painting'])
    parser.add_argument('--target-domain', nargs='+', default=['sketch'])
    parser.add_argument('--known-classes', nargs='+',
                        default=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house', 'person'])
    parser.add_argument('--unknown-classes', nargs='+', default=[])

    """OfficeH"""
    # parser.add_argument('--dataset', default='OfficeHome')
    # parser.add_argument('--source-domain', nargs='+', default=['Art', 'Clipart', 'Product'])
    # parser.add_argument('--target-domain', nargs='+', default=['RealWorld'])
    # parser.add_argument('--known-classes', nargs='+', default=['Alarm_Clock', 'Backpack', 'Batteries', ...])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    # parser.add_argument('--dataset', default='DigitsDG')
    # parser.add_argument('--source-domain', nargs='+', default=['mnist', 'mnist_m', 'svhn'])
    # parser.add_argument('--target-domain', nargs='+', default=['syn'])
    # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4', '5'])
    # parser.add_argument('--unknown-classes', nargs='+', default=['6', '7', '8', '9'])

    """VLCS"""
    # parser.add_argument('--dataset', default='VLCS')
    # parser.add_argument('--source-domain', nargs='+', default=['CALTECH', 'PASCAL', 'SUN'])
    # parser.add_argument('--target-domain', nargs='+', default=['LABELME',])
    # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4'])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    """TerraIncognita"""
    # parser.add_argument('--dataset', default='TerraIncognita')
    # parser.add_argument('--source-domain', nargs='+', default=['location_38', 'location_43', 'location_46'])
    # parser.add_argument('--target-domain', nargs='+', default=['location_100'])
    # parser.add_argument('--known-classes', nargs='+', default=['bobcat', 'coyote', 'dog', 'opossum', 'rabbit', 'raccoon', 'squirrel', 'bird', 'cat', 'empty',])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    """DomainNet"""
    # parser.add_argument('--dataset', default='DomainNet')
    # parser.add_argument('--source-domain', nargs='+', default=['clipart', 'infograph', 'painting', 'quickdraw', 'real'])
    # parser.add_argument('--target-domain', nargs='+', default=['sketch'])
    # parser.add_argument('--known-classes', nargs='+', default=['aircraft_carrier', 'airplane', ...])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    parser.add_argument('--random-split', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--algorithm', default='arith')
    parser.add_argument('--task-d', type=int, default=3)
    parser.add_argument('--task-c', type=int, default=3)
    parser.add_argument('--task-per-step', nargs='+', type=int, default=[3, 3, 3])
    parser.add_argument('--weight-per-step', nargs='+', type=float, default=[1.5, 1, 0.5], help='arith only')
    parser.add_argument('--selection-mode', default='random')  # random, hard
    parser.add_argument('--arith-antithetic', action='store_true', help = 'Alternate domain order by epoch (π on even, π_rev on odd). Mirror step schedules accordingly.')

    parser.add_argument('--net-name', default='resnet50')
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=6000)
    parser.add_argument('--eval-step', type=int, default=300)
    parser.add_argument('--lr', type=float,
                        default=2e-4)  # Alpha (meta-lr) has been calculated in the following code, so it is set to 1/t of the default learning rate.
    parser.add_argument('--meta-lr', type=float, default=1e-2)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-cls', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')

    parser.add_argument('--save-dir', default='/data/changsik/arith/save')
    parser.add_argument('--save-name', default='demo')
    parser.add_argument('--save-best-test', action='store_true')
    parser.add_argument('--save-later', action='store_true')

    parser.add_argument('--num-epoch-before', type=int, default=0)

    return parser.parse_args()


args = get_args()


# It can be used to replace the following code, but the editor may take it as an error.
# locals().update(vars(args))

# It can be replaced by the preceding code.
dataset = args.dataset
source_domain = sorted(args.source_domain)
target_domain = sorted(args.target_domain)
known_classes = sorted(args.known_classes)
unknown_classes = sorted(args.unknown_classes)
random_split = args.random_split
seed = args.seed
gpu = args.gpu
batch_size = args.batch_size
algorithm = args.algorithm
task_d = args.task_d
task_c = args.task_c
task_per_step = args.task_per_step
weight_per_step = args.weight_per_step
selection_mode = args.selection_mode
net_name = args.net_name
optimize_method = args.optimize_method
schedule_method = args.schedule_method
num_epoch = args.num_epoch
eval_step = args.eval_step
lr = args.lr
meta_lr = args.meta_lr
nesterov = args.nesterov
without_cls = args.without_cls
without_bcls = args.without_bcls
share_param = args.share_param
save_dir = args.save_dir
save_name = args.save_name
save_later = args.save_later
save_best_test = args.save_best_test
num_epoch_before = args.num_epoch_before

arith_antithetic = args.arith_antithetic
crossval = True

if dataset == 'PACS':
    train_dir = '/data/datasets/domainbed/PACS'
    val_dir = '/data/datasets/domainbed/PACS'
    test_dir = '/data/datasets/domainbed/PACS'
    sub_batch_size = batch_size // 2
    small_img = False
elif dataset == 'OfficeHome':
    train_dir = '/data/datasets/domainbed/OfficeH'
    val_dir = '/data/datasets/domainbed/OfficeH'
    test_dir = '/data/datasets/domainbed/OfficeH'
    sub_batch_size = batch_size // 2
    small_img = False
elif dataset == "DigitsDG":
    train_dir = ''
    val_dir = ''
    test_dir = ''
    sub_batch_size = batch_size // 2
    small_img = True
elif dataset == 'VLCS':
    train_dir = '/data/datasets/domainbed/VLCS'
    val_dir = '/data/datasets/domainbed/VLCS'
    test_dir = '/data/datasets/domainbed/VLCS'
    sub_batch_size = batch_size
    small_img = False
elif dataset == 'TerraIncognita':
    train_dir = '/data/datasets/domainbed/TerraInc'
    val_dir = '/data/datasets/domainbed/TerraInc'
    test_dir = '/data/datasets/domainbed/TerraInc'
    sub_batch_size = batch_size
    small_img = False
elif dataset == "DomainNet":
    train_dir = '/data/datasets/domainbed/DomainNet'
    val_dir = '/data/datasets/domainbed/DomainNet'
    test_dir = '/data/datasets/domainbed/DomainNet'
    sub_batch_size = batch_size // 2
    small_img = False

log_path = os.path.join(save_dir, 'log', save_name + '_train.txt')
param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
model_val_path = os.path.join(save_dir, 'model', 'val', save_name + '.tar')
model_test_path = os.path.join(save_dir, 'model', 'test', save_name + '.tar')
renovate_step = int(num_epoch * 0.85) if save_later else 0

assert task_d * task_c == sum(task_per_step)

os.makedirs(os.path.dirname(log_path), exist_ok=True)
os.makedirs(os.path.dirname(param_path), exist_ok=True)
os.makedirs(os.path.dirname(model_val_path), exist_ok=True)
os.makedirs(os.path.dirname(model_test_path), exist_ok=True)


