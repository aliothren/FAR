import torch
import argparse
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

###### ----------- Global PATHs and configs ----------- ######
# Path to datasets
DATA_PATH = {
    "IMNET": "",
    "CIFAR100": "/home/yuxinr/far/data/cifar100",
    "CIFAR10": "",
    "FLOWER": "",
    "CAR": "",
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/home/yuxinr/far/FAR/"


###### ----------- Shared parsers for MX100 deploy ----------- ######
def get_common_parser():
    parser = argparse.ArgumentParser("parser for basic environment and dataset", add_help=False)
    
    # Environment setups
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base output directory")
    parser.add_argument("--output-dir", default='', help="Output path, do NOT change here")
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    
    # Dataset parameters
    parser.add_argument("--input-size", default=224, type=int, help="expected images size for model input")
    parser.add_argument("--dataset", default="CIFAR100", type=str, 
                        choices=["IMNET", "CIFAR10", "CIFAR100", "INAT18", "INAT19", "FLOWER", "CAR"])
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument("--data-path", default="", type=str, help="Path of dataset")
    parser.add_argument("--nb-classes", default=1000, type=int, 
                        help="Number of classes in dataset (default:1000)")
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")

    return parser


def fill_default_common_args(args):
    if args.data_path == "":
        args.data_path = DATA_PATH[args.dataset]
        print(f"Using default dataset path {args.data_path}")
    return args


def build_transform(args):
    resize_im = args.input_size > 32
    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def load_onnx_val_dataset(args):
    print(f"Loading validation dataset {args.dataset}")
    transform = build_transform(args)

    if args.dataset == "CIFAR100":
        dataset_val = datasets.CIFAR100(args.data_path, train=False, transform=transform)
        
    elif args.dataset == "CIFAR10":
        dataset_val = datasets.CIFAR10(args.data_path, train=False, transform=transform)
    
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader_val


def evaluate_onnx_model(model_path, val_loader): 
    
    # ort_session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
    ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in val_loader:
        images_np = images.numpy() if isinstance(images, torch.Tensor) else images
        ort_outputs = ort_session.run([output_name], {input_name: images_np})[0]  # shape: [batch, num_classes]

        top1_preds = np.argmax(ort_outputs, axis=1)
        top5_preds = np.argsort(ort_outputs, axis=1)[:, -5:]

        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

        top1_correct += np.sum(top1_preds == labels_np)
        top5_correct += np.sum([label in top5 for label, top5 in zip(labels_np, top5_preds)])
        total += labels_np.shape[0]

    print(f"Accuracy on ONNX model:  Acc@1 {100 * top1_correct / total:.2f}%, Acc@5 {100 * top5_correct / total:.2f}%")

