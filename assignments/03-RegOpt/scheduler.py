from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    It is a scheduler
    """

    def __init__(
        self,
        optimizer,
        start_factor=1.0 / 3,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
        verbose=False,
    ):
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        """
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """
        Learning rate schedule
        Arguments:
        None
        Returns:
            the scheduler

        """
        if self.last_epoch < 2500:
            return [base_lr for base_lr in self.base_lrs]

        if self.last_epoch < 6000:
            return [
                base_lr * (1 - (self.last_epoch - 2500) / 6000)
                for base_lr in self.base_lrs
            ]

        return [
            base_lr * (0.9999 ** (self.last_epoch - 6000)) for base_lr in self.base_lrs
        ]
