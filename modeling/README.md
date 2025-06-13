# Is Attention Required for Transformer Inference? Explore Function-preserving Attention Replacement

This repository contains the official implementation of our paper:
**Is Attention Required for Transformer Inference? Explore Function-preserving Attention Replacement**.

> This work explores a plug-in replacement for attention using multi-head bidirectional LSTMs with block-level distillation and structured sparsity.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

We recommend Python 3.8+ with PyTorch ≥ 1.13. Training was tested on 8× A6000 GPUs and AMD EPYC 9254 CPUs.

## Training

To train FAR models with multi-head LSTM replacing attention blocks:

### DeiT-Base:

```bash
torchrun --nproc_per_node=4 --master-port=<PORT> main.py \
  --distributed --mode train --rep-by multi-lstm \
  --base-model DeiT-Base --batch-size 64 --scale Base \
  --block-ft-lr 2e-5 --block-ft-batch-size 64 --skip-train-attn
```

### DeiT-Small:

```bash
torchrun --nproc_per_node=4 --master-port=<PORT> main.py \
  --distributed --mode train --rep-by multi-lstm \
  --base-model DeiT-Small --batch-size 64 --scale Small \
  --block-ft-lr 2e-5 --block-ft-batch-size 64 --skip-train-attn
```

### DeiT-Tiny:

```bash
python3 main.py --mode train --rep-by multi-lstm
```

## Pruning

To perform structured pruning on a trained model:

### Multi-GPU:

```bash
torchrun --nproc_per_node=4 --master-port=<PORT> prune.py \
  --distributed --batch-size 64 --scale <Tiny|Small|Base> \
  --model-path <PATH_TO_MODEL>
```

### Single GPU:

```bash
python3 prune.py --scale <Tiny|Small|Base>
```

## Downstream Finetuning

Set `PRETRAINED_PATH_PRUNE` in `main.py` before launching:

```bash
python3 main.py --mode downstream --scale <Tiny|Small|Base> --dataset <DATASET> \
  --lr 5e-5 --epochs 1000 --reprob 0.0 --drop-path 0.0
```

To finetune a pruned model:

Set `PRETRAINED_PATH` in `main.py` and run:

```bash
python3 main.py --mode downstream --ds-pruned --scale <Tiny|Small|Base> \
  --dataset <DATASET> --lr 5e-5 --epochs 1000 --reprob 0.0 --drop-path 0.0
```

## Visualization

To visualize token importance maps:

```bash
python3 visualization.py
```

## Acknowledgements

This repository is built upon the official implementations of:

* DeiT: [https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)
* DeepHoyer: [https://github.com/yanghr/DeepHoyer](https://github.com/yanghr/DeepHoyer)

We thank the original authors for their contributions.
