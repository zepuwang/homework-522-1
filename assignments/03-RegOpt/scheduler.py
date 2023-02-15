from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    It is a scheduler
    """

    def __init__(
        self, optimizer, total_iters=10, power=1.0, last_epoch=-1, verbose=False
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """
        Learning rate schedule

        Arguments:
           None

        Returns:
            the scheduler
        """
        decay_factor = 0
        if (1.0 - (self.last_epoch - 1) / self.total_iters) != 0:
            decay_factor = (
                (1.0 - self.last_epoch / self.total_iters)
                / (1.0 - (self.last_epoch - 1) / self.total_iters)
            ) ** self.power
        if (1.0 - (self.last_epoch - 1) / self.total_iters) == 0:
            decay_factor = 1
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]
