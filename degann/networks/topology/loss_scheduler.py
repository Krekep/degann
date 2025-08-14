class LossScheduler:
    """
    Simple loss scheduler. First k epochs it returns only boundary and initial loss indices, after returns all loss indices
    """

    def __init__(
        self,
        k: int,
        boundary_indices: list[int],
        loss_weights: list[float],
        **kwargs,
    ):
        self.k = k
        self.current_step = 0
        self.boundary_indices = boundary_indices
        self.all_indices = boundary_indices + [len(boundary_indices)]
        self.loss_weights = loss_weights

    def on_epoch_start(self, **kwargs) -> tuple[list[int], list[float]]:
        if self.current_step <= self.k:
            return self.boundary_indices, self.loss_weights
        return self.all_indices, self.loss_weights

    def step(self):
        self.current_step += 1


class AdaptiveLossScheduler(LossScheduler):
    def __init__(
        self,
        k: int,
        boundary_indices: list[int],
        loss_weights: list[float],
        threshold: float = 1e-3,
        loss_multiplier: float = 10,
        **kwargs,
    ):
        super().__init__(k, boundary_indices, loss_weights)
        self.threshold = threshold
        self.loss_multiplier = loss_multiplier

    def on_epoch_start(
        self, curr_max_w_update, **kwargs
    ) -> tuple[list[int], list[float]]:
        if (
            curr_max_w_update < self.threshold
            and max(self.loss_weights) < 100_000
            and self.current_step > self.k
        ):
            self.loss_weights = list(
                map(lambda x: x * self.loss_multiplier, self.loss_weights)
            )
            print(f"Update loss weights {self.loss_weights}, step {self.current_step}")
        if self.current_step <= self.k:
            return self.boundary_indices, self.loss_weights
        return self.all_indices, self.loss_weights

    def step(self):
        self.current_step += 1
