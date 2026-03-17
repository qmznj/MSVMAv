# MSVMAv: Official Implementation

This repository provides the **official implementation** of the MSVMAv algorithm proposed in:

> **On the Optimization of Margin Distribution**  
> *IJCAI 2022*

## Overview

MSvMAv is a margin-distribution-based optimization method that improves generalization performance by alternatively optimizing:
- the **mean** of margins, and  
- the **variance** of margins.

# Quick Start:
```
# install dependencies
pip install numpy tqdm numba scikit-learn

# run the demo
python MSVMAv.py
```

# Citation

If you find this work useful, please cite:
```
@inproceedings{QianAZG22,
  author       = {Meng{-}Zhang Qian and Zheng Ai and Teng Zhang and Wei Gao},
  title        = {On the Optimization of Margin Distribution},
  booktitle    = {Proceedings of the Thirty-First International Joint Conference on
                  Artificial Intelligence, 2022},
  pages        = {3387--3393},
  year         = {2022},
}
```
