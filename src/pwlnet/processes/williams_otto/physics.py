"""Williams-Otto physics-informed training terms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pwlnet.processes.williams_otto.model import WilliamsOttoParameters
from pwlnet.training.physics_informed import PhysicsLoss, register_physics

__all__ = ["WilliamsOttoPhysics"]


class WilliamsOttoPhysics(PhysicsLoss):
    """Williams-Otto mass-fraction closure and operating-bound penalties."""

    name = "williams_otto"
    input_dim = 2
    output_dim = 6
    default_weights = {"closure": 0.0, "bounds": 0.0}

    def __init__(self, params: WilliamsOttoParameters | None = None) -> None:
        self.params = params if params is not None else WilliamsOttoParameters()

    def build_constants(self, tf) -> dict[str, Any]:
        p = self.params
        return {
            "xa_max": tf.constant(p.xa_max, dtype=tf.float32),
            "xg_max": tf.constant(p.xg_max, dtype=tf.float32),
        }

    def loss_terms(self, tf, x_orig, y_orig, constants: Mapping[str, Any]) -> dict[str, Any]:
        del x_orig
        c = constants
        xa = y_orig[:, 0]
        xg = y_orig[:, 4]
        return {
            "closure_loss": tf.reduce_mean(tf.square(tf.reduce_sum(y_orig, axis=1) - 1.0)),
            "bound_loss": tf.reduce_mean(
                tf.square(tf.nn.relu(xa - c["xa_max"]))
                + tf.square(tf.nn.relu(xg - c["xg_max"]))
            ),
        }


register_physics(WilliamsOttoPhysics.name, WilliamsOttoPhysics)
