This repository containes the full implementation of FAR.

## Structure
```bash
    /benchmodels
        default models

    /checkpoints
        all models, checkpoints, logs and figures for FAR model training
    
    /deploy
        hardware implementation of FAR models

    /hw_files
        models for hardware implementation

    /modeling
        software implementation of FAR models
        including architechture design, training, evaluating, pruning, visualization, etc.
```


## Acknowledgements

This repository is built upon the official implementations of:

* DeiT: [https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)
* DeepHoyer: [https://github.com/yanghr/DeepHoyer](https://github.com/yanghr/DeepHoyer)
* Tetramem software team (Non-public)

We thank the original authors for their contributions.