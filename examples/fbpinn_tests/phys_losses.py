import tensorflow as tf


class PhysLoss:
    def  __init__(self, description: str = ""):
        description = "du/dx = omega * cos(omega*x), y(0) = 0"
        self.description = description
        self.full_losses: list[callable] = []  # on full domain
        self.sub_losses: list[tuple[callable, dict[str, tuple[float, float]]]] = []  # on subdomain (e.g. [a; b] \in D)
