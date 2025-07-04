import tensorflow as tf


class PhysLoss:
    def __init__(self, description: str = ""):
        self.description = description
        self.full_losses: list[callable] = []  # on full domain
        self.sub_losses: list[
            tuple[callable, dict[str, tuple[float, float]]]
        ] = []  # on subdomain (e.g. [a; b] \in D)
