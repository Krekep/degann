class BaseLayerScheduler:
    def __init__(self, n: int):
        self.current_indices = list(range(0, n))

    def on_epoch_start(self) -> list[int]:
        return self.current_indices

    def get_frozen_indices(self) -> list[int]:
        return []

    def step(self):
        pass


class SequenceLayerScheduler(BaseLayerScheduler):
    def __init__(
        self,
        n: int,
        left_bound_step: int,
        right_bound_step: int,
        left_bound_schedule: int,
        right_bound_schedule: int,
        start_left_bound: int,
        start_right_bound: int,
    ):
        super().__init__(n=n)
        self.n = n
        self.left_bound_step = left_bound_step
        self.right_bound_step = right_bound_step
        self.left_bound_schedule = left_bound_schedule
        self.right_bound_schedule = right_bound_schedule
        self.current_step = 0
        self.current_left_bound = start_left_bound
        self.current_right_bound = start_right_bound
        self.current_indices = list(range(start_left_bound, start_right_bound))
        self.is_changes = True
        self.frozen_indices = []

    def on_epoch_start(self) -> list[int]:
        if self.is_changes:
            range_changed = False
            if self.current_step % self.right_bound_schedule == 0:
                if self.current_right_bound == self.n:
                    self.current_left_bound = 0
                    self.is_changes = False
                self.current_right_bound = min(
                    self.n, self.current_right_bound + self.right_bound_step
                )
                range_changed = True

            active_lb = self.current_left_bound
            if self.is_changes and self.current_step % self.left_bound_schedule == 0:
                curr_lb = self.current_left_bound
                new_lb = self.current_left_bound + self.left_bound_step
                self.current_left_bound = new_lb
                self.frozen_indices = [i for i in range(curr_lb, new_lb)]
                range_changed = True
            if range_changed:
                self.current_indices = list(range(active_lb, self.current_right_bound))
        res = self.current_indices
        return res

    def get_frozen_indices(self) -> list[int]:
        return self.frozen_indices

    def step(self):
        self.current_step += 1


class ReduceEpochsLayerScheduler(SequenceLayerScheduler):
    def __init__(
        self, reduce_step: int, reduce_schedule: int, reduce_count: int, **kwargs
    ):
        super().__init__(**kwargs)
        self.reduce_step = reduce_step
        self.reduce_schedule = reduce_schedule
        self.reduce_count = reduce_count

    def on_epoch_start(self) -> list[int]:
        if self.reduce_count > 0 and self.current_step % self.reduce_schedule == 0:
            self.left_bound_schedule -= self.reduce_step
            self.right_bound_schedule -= self.reduce_step
            self.reduce_count -= 1

        return super().on_epoch_start()
