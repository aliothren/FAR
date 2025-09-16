import argparse
import numpy as np
import pathlib
from cifar100 import CIFAR100Dataset


def _get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar-dir", type=str, required=True, help="")
    parser.add_argument("--split", type=str, default="test", choices=["train","test"])
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument(
        "--normalize", 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help=""
    )
    parser.add_argument("--num-samples", type=int, default=1, help="")
    parser.add_argument("--out-dir", type=str, required=True, help="")
    parser.add_argument(
        "--save-labels", 
        action=argparse.BooleanOptionalAction, 
        default=True, 
        help=""
    )
    return parser


def main():
    args = _get_args_parser().parse_args()

    dataset = CIFAR100Dataset(
        cifar_dir=args.cifar_dir,
        split=args.split,
        output_size=args.output_size,
        normalize=args.normalize,
    )

    images, labels = dataset.all_images_and_labels() 

    if args.num_samples > 0:
        n = min(args.num_samples, images.shape[0])
        images = images[:n]
        labels = labels[:n]

    out_dir = pathlib.Path(args.out_dir)
    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "input" / "input.npy", images)
    if args.save_labels:
        np.save(out_dir / "labels.npy", labels)

    print(f"Saved images to: {out_dir / 'input' / 'input.npy'}  \
            shape={images.shape} dtype={images.dtype}")
    if args.save_labels:
        print(f"Saved labels to: {out_dir / 'labels.npy'}  \
              shape={labels.shape} dtype={labels.dtype}")

if __name__ == "__main__":
    main()
