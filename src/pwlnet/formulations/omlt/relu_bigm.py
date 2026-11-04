"""ReLU big-M OMLT formulation factory."""

from __future__ import annotations

from typing import Any

__all__ = ["relu_bigm_formulation"]


def relu_bigm_formulation(network_structure: Any) -> Any:
    """Build an OMLT ReLU big-M formulation.

    Parameters
    ----------
    network_structure : omlt.neuralnet.NetworkDefinition
        OMLT network structure produced by
        :func:`pwlnet.formulations.omlt.to_omlt_network`.

    Returns
    -------
    omlt.neuralnet.ReluBigMFormulation
        Formulation object ready for ``OmltBlock.build_formulation``.
    """
    from omlt.neuralnet import ReluBigMFormulation

    return ReluBigMFormulation(network_structure)
