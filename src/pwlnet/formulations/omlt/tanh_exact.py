"""Exact tanh OMLT formulation factory."""

from __future__ import annotations

from typing import Any

__all__ = ["tanh_exact_formulation"]


def tanh_exact_formulation(network_structure: Any) -> Any:
    """Build an OMLT smooth full-space formulation for tanh networks.

    Parameters
    ----------
    network_structure : omlt.neuralnet.NetworkDefinition
        OMLT network structure produced by
        :func:`pwlnet.formulations.omlt.to_omlt_network`.

    Returns
    -------
    omlt.neuralnet.FullSpaceSmoothNNFormulation
        Formulation object ready for ``OmltBlock.build_formulation``.
    """
    from omlt.neuralnet import FullSpaceSmoothNNFormulation

    return FullSpaceSmoothNNFormulation(network_structure)
