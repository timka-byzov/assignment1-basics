import numpy as np


def get_lr_cosine_schedule(
    step: int, max_lr: float, min_lr: float, T_warmup, T_max
) -> float:
    if step < T_warmup:
        return 0 + step / T_warmup * max_lr
    elif T_warmup <= step <= T_max:
        step -= T_warmup
        T_max -= T_warmup
        return min_lr + (1 + np.cos(step * np.pi / T_max)) * (max_lr - min_lr) / 2
    else:
        return min_lr
