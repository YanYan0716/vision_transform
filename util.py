import numpy as np


import config


class LearningRateSchedule(object):
    def __init__(self, total_step, base, decay_type, warmup_steps, linear_end=1e-5):
        self.total_step = total_step
        self.base = base
        self.decay_type = decay_type
        self.warmup_steps = warmup_steps
        self.linear_end = linear_end

    def __call__(self, step):
        lr = self.base
        progress = (step - self.warmup_steps) / float(self.total_step - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        if self.decay_type == 'linear':
            lr = self.linear_end +(lr - self.linear_end)*(1.0 - progress)
        elif self.decay_type == 'cosine':
            lr = lr*0.5*(1. + np.cos(np.pi * progress))
        else:
            raise ValueError(f'Unknown lr type {self.decay_type}')

        if self.warmup_steps:
            lr = lr*np.minimum(1., step/self.warmup_steps)

        return np.asarray(lr, dtype=np.float32)


def test():
    lr_schedule = LearningRateSchedule(
        total_step=config.TOTAL_STEPS,
        base=config.BASE_LR,
        decay_type=config.DECAY_TYPE,
        warmup_steps=config.WARMUP_STEPS
    )

    for i in range(5):
        lr = lr_schedule.__call__(i)
        print(lr)


if __name__ == '__main__':
    test()