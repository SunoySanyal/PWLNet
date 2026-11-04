"""Construct surrogate blocks on existing Pyomo models.

This module turns a :class:`pwlnet.core.network.NetworkDefinition` into
a surrogate block attached to a user-supplied Pyomo model. The preferred
OMLT conversion path is the project exporter
:mod:`pwlnet.formulations.omlt` (``to_omlt_network`` /
``build_omlt_formulation``); when that module is unavailable (or its
API differs), a self-contained fallback conversion is used so embedding
never depends on import order.

pyomo and omlt are imported lazily inside functions; importing this
module never requires either package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.core.network import NetworkDefinition
from pwlnet.utils.exceptions import FormulationError
from pwlnet.utils.logging import get_logger

__all__ = [
    "SUPPORTED_FORMULATIONS",
    "resolve_omlt_network",
    "build_omlt_formulation",
    "build_surrogate_block",
]

_logger = get_logger(__name__)

SUPPORTED_FORMULATIONS: tuple[str, ...] = (
    "relu_bigm",
    "tanh_exact",
    "relu_partition",
    "relu_complementarity",
    "tanh_pwl_sos2",
    "tanh_pwl_bigm",
)
"""Formulation names accepted by :func:`build_surrogate_block`."""

_RELU_FORMULATIONS = frozenset(
    {"relu_bigm", "relu_partition", "relu_complementarity"}
)


def _sibling_exporter() -> Any | None:
    """Import the :mod:`pwlnet.formulations.omlt` exporter if present.

    The exporter is developed independently; this helper degrades to
    ``None`` when the module is missing or fails to import, so callers
    can fall back to the local conversion.
    """
    try:
        from pwlnet.formulations import omlt as exporter
    except Exception as exc:  # pragma: no cover - environment dependent
        _logger.debug("pwlnet.formulations.omlt not importable (%s).", exc)
        return None
    return exporter


def _validate_formulation_choice(network: NetworkDefinition, formulation: str) -> None:
    """Check that ``formulation`` is known and compatible with the network.

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If the formulation name is unknown, a ReLU-based formulation is
        requested for a network containing tanh layers, or the smooth
        ``"tanh_exact"`` formulation is requested for a network
        containing ReLU layers.
    """
    if formulation not in SUPPORTED_FORMULATIONS:
        raise FormulationError(
            f"Unknown formulation '{formulation}'. "
            f"Supported: {list(SUPPORTED_FORMULATIONS)}."
        )
    activations = set(network.activations)
    if formulation in _RELU_FORMULATIONS and "tanh" in activations:
        raise FormulationError(
            f"Formulation '{formulation}' requested but the network has tanh "
            "layers; ReLU MIP formulations are only exact for piecewise-"
            "linear activations. Use 'tanh_exact' instead."
        )
    if formulation == "tanh_exact" and "relu" in activations:
        raise FormulationError(
            "Formulation 'tanh_exact' (smooth full-space NLP) requested but "
            "the network has ReLU layers, which are not smooth. Use "
            "'relu_bigm' (or another ReLU MIP formulation) instead."
        )
    if formulation in {"tanh_pwl_sos2", "tanh_pwl_bigm"}:
        hidden = set(network.activations[:-1])
        if hidden - {"tanh"}:
            raise FormulationError(
                f"Formulation '{formulation}' requires tanh hidden layers."
            )


def _to_omlt_network_fallback(network: NetworkDefinition) -> Any:
    """Convert a PWLNet network to an OMLT ``NetworkDefinition``.

    Self-contained fallback used when the
    :mod:`pwlnet.formulations.omlt` exporter is unavailable. Scaling is
    delegated to ``omlt.scaling.OffsetScaling`` (same convention as
    :class:`pwlnet.core.scaling.OffsetScaler`:
    ``x_scaled = (x - offset) / factor``), so the resulting OMLT block
    exposes ``inputs``/``outputs`` in ORIGINAL (unscaled) units.

    Parameters
    ----------
    network : NetworkDefinition
        PWLNet network with ``input_bounds`` set (required by OMLT for
        big-M and bound tightening).

    Returns
    -------
    omlt.neuralnet.NetworkDefinition
        The equivalent OMLT network structure.

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If omlt is not installed or the network has no input bounds.
    """
    try:
        from omlt.neuralnet import NetworkDefinition as OmltNetworkDefinition
        from omlt.neuralnet.layer import DenseLayer as OmltDenseLayer
        from omlt.neuralnet.layer import InputLayer as OmltInputLayer
        from omlt.scaling import OffsetScaling
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise FormulationError(
            "omlt is required to build surrogate blocks; install it with "
            "'pip install omlt'."
        ) from exc

    if network.input_bounds is None:
        raise FormulationError(
            "NetworkDefinition.input_bounds is None; OMLT formulations need "
            "input bounds (big-M values and bound tightening). Set "
            "input_bounds on the network before embedding."
        )

    scaled_bounds = {
        i: (float(lb), float(ub))
        for i, (lb, ub) in enumerate(network.scaled_input_bounds())
    }

    scaling = None
    if network.input_scaler is not None or network.output_scaler is not None:
        d, m = network.n_inputs, network.n_outputs
        if network.input_scaler is not None:
            in_offset = np.asarray(network.input_scaler.offset, dtype=float)
            in_factor = np.asarray(network.input_scaler.factor, dtype=float)
        else:
            in_offset, in_factor = np.zeros(d), np.ones(d)
        if network.output_scaler is not None:
            out_offset = np.asarray(network.output_scaler.offset, dtype=float)
            out_factor = np.asarray(network.output_scaler.factor, dtype=float)
        else:
            out_offset, out_factor = np.zeros(m), np.ones(m)
        scaling = OffsetScaling(
            offset_inputs=[float(v) for v in in_offset],
            factor_inputs=[float(v) for v in in_factor],
            offset_outputs=[float(v) for v in out_offset],
            factor_outputs=[float(v) for v in out_factor],
        )

    omlt_net = OmltNetworkDefinition(
        scaling_object=scaling, scaled_input_bounds=scaled_bounds
    )
    prev = OmltInputLayer([network.n_inputs])
    omlt_net.add_layer(prev)
    for layer in network.layers:
        dense = OmltDenseLayer(
            [layer.n_inputs],
            [layer.n_outputs],
            weights=np.asarray(layer.weights, dtype=float),
            biases=np.asarray(layer.biases, dtype=float),
            activation=layer.activation,  # "relu"/"tanh"/"linear" are OMLT names
        )
        omlt_net.add_layer(dense)
        omlt_net.add_edge(prev, dense)
        prev = dense
    return omlt_net


def resolve_omlt_network(network: NetworkDefinition) -> Any:
    """Get an OMLT ``NetworkDefinition`` for a PWLNet network.

    Prefers ``pwlnet.formulations.omlt.to_omlt_network`` (the project
    exporter) and falls back to a local conversion when the exporter is
    absent or its signature differs.

    Parameters
    ----------
    network : NetworkDefinition
        The PWLNet network to convert.

    Returns
    -------
    omlt.neuralnet.NetworkDefinition
        The OMLT network structure.
    """
    exporter = _sibling_exporter()
    if exporter is not None:
        fn = getattr(exporter, "to_omlt_network", None)
        if callable(fn):
            try:
                return fn(network)
            except TypeError as exc:
                _logger.warning(
                    "pwlnet.formulations.omlt.to_omlt_network(network) failed "
                    "(%s); using the embedding fallback conversion.",
                    exc,
                )
    return _to_omlt_network_fallback(network)


def build_omlt_formulation(
    network: NetworkDefinition, formulation: str = "relu_bigm"
) -> Any:
    """Build an OMLT formulation object for a PWLNet network.

    Prefers ``pwlnet.formulations.omlt.build_omlt_formulation`` /
    ``get_omlt_formulation`` from the project exporter; otherwise the
    network is converted with :func:`resolve_omlt_network` and the
    matching ``omlt.neuralnet`` formulation class is instantiated.

    Parameters
    ----------
    network : NetworkDefinition
        The PWLNet network to formulate.
    formulation : str, optional
        One of :data:`SUPPORTED_FORMULATIONS`:

        * ``"relu_bigm"`` -> ``ReluBigMFormulation`` (MILP)
        * ``"tanh_exact"`` -> ``FullSpaceSmoothNNFormulation`` (NLP)
        * ``"relu_partition"`` -> ``ReluPartitionFormulation``
        * ``"relu_complementarity"`` -> ``ReluComplementarityFormulation``

    Returns
    -------
    object
        An OMLT formulation ready for ``OmltBlock.build_formulation``.

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If the formulation is unknown or incompatible with the
        network's activations, or omlt is not installed.
    """
    _validate_formulation_choice(network, formulation)

    exporter = _sibling_exporter()
    if exporter is not None:
        for fn_name in ("build_omlt_formulation", "get_omlt_formulation"):
            fn = getattr(exporter, fn_name, None)
            if callable(fn):
                try:
                    return fn(network, formulation)
                except TypeError as exc:
                    _logger.warning(
                        "pwlnet.formulations.omlt.%s(network, formulation) "
                        "failed (%s); using the embedding fallback.",
                        fn_name,
                        exc,
                    )

    omlt_net = resolve_omlt_network(network)
    try:
        import omlt.neuralnet as omlt_nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise FormulationError(
            "omlt is required to build surrogate blocks; install it with "
            "'pip install omlt'."
        ) from exc
    formulation_classes = {
        "relu_bigm": omlt_nn.ReluBigMFormulation,
        "tanh_exact": omlt_nn.FullSpaceSmoothNNFormulation,
        "relu_partition": omlt_nn.ReluPartitionFormulation,
        "relu_complementarity": omlt_nn.ReluComplementarityFormulation,
    }
    return formulation_classes[formulation](omlt_net)


def build_surrogate_block(
    model: Any,
    network: NetworkDefinition,
    formulation: str = "relu_bigm",
    block_name: str = "surrogate",
    **formulation_options: Any,
) -> Any:
    """Attach a surrogate block to an existing Pyomo model.

    Adds a block named ``block_name`` to ``model`` and builds the
    requested formulation on it. The block's ``inputs`` and ``outputs``
    variables are in ORIGINAL (unscaled) units when the network carries
    scalers.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel or Block
        The model to attach the surrogate block to.
    network : NetworkDefinition
        The trained PWLNet network (must have ``input_bounds``).
    formulation : str, optional
        See :func:`build_omlt_formulation`. Also accepts
        ``"tanh_pwl_sos2"`` and ``"tanh_pwl_bigm"`` for the manual
        Pyomo PWL-tanh formulations.
    block_name : str, optional
        Attribute name for the new block. Default ``"surrogate"``.

    Returns
    -------
    pyomo Block or omlt.OmltBlock
        The constructed block (``getattr(model, block_name)``).

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If required formulation packages are not installed, the model
        already has a component named ``block_name``, or the formulation
        is incompatible with the network.
    """
    formulation = str(formulation).strip().lower()
    _validate_formulation_choice(network, formulation)

    if formulation in {"tanh_pwl_sos2", "tanh_pwl_bigm"}:
        from pwlnet.formulations.pyomo_pwl import (
            build_tanh_pwl_bigm_block,
            build_tanh_pwl_sos2_block,
        )

        builder = (
            build_tanh_pwl_sos2_block
            if formulation == "tanh_pwl_sos2"
            else build_tanh_pwl_bigm_block
        )
        return builder(
            model,
            network,
            block_name=block_name,
            breakpoints=formulation_options.get("breakpoints", 5),
        )

    try:
        from omlt import OmltBlock  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise FormulationError(
            "omlt is required to build surrogate blocks; install it with "
            "'pip install omlt'."
        ) from exc

    if model.component(block_name) is not None:
        raise FormulationError(
            f"Model already has a component named '{block_name}'; pass a "
            "different block_name."
        )

    formulation_obj = build_omlt_formulation(network, formulation)
    setattr(model, block_name, OmltBlock())
    block = getattr(model, block_name)
    block.build_formulation(formulation_obj)
    _logger.info(
        "Built OMLT block '%s' (%s) for network %r.",
        block_name,
        formulation,
        network,
    )
    return block
