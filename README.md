# Ranked Entropy Minimization for Continual Test-Time Adaptation
<div style="text-align: center;">
  <a href="https://arxiv.org/abs/2505.16441"><img src="https://img.shields.io/badge/arXiv-2505.16441-b31b1b.svg" alt="arXiv"></a>
</div>

Welcome to the official code repository for Ranked Entropy Minimization for Continual Test-Time Adaptation (ICML 2025)

## Instructions
Instructions will be added soon.
This is an early access release, we are currently refactoring the code and verifying its reproducibility.

```bash
cd imagenet
CUDA_VISIBLE_DEVICES=0 python imagenetc.py --cfg cfgs/vit/rem.yaml --data_dir <your_data_path>
```

## Acknowledgement 
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ ViDA [official](https://github.com/Yangsenqiao/vida)
+ Continual-MAE [official](https://github.com/RanXu2000/continual-mae)
+ MaskedKD [official](https://github.com/effl-lab/MaskedKD)
+ KATANA [official](https://github.com/giladcohen/KATANA) 
+ Robustbench [official](https://github.com/RobustBench/robustbench) 
