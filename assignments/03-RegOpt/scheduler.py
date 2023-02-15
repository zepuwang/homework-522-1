from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    It is a scheduler
    """

    def __init__(self, optimizer, step_size=0.1, gamma=0.1, last_epoch=-1, verbose=False):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.step_size = step_size
        self.gamma = gamma
        super(_LRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """
        Learning rate schedule

        Arguments:
           None

        Returns:
            the scheduler
        """
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]
