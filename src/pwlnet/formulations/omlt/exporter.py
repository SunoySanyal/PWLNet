"""Export PWLNet networks to OMLT network definitions and Pyomo blocks.

This module bridges :class:`pwlnet.core.network.NetworkDefinition` (pure
numpy) and OMLT's optimization formulations. The OMLT network is built
DIRECTLY from our layers -- no round-trip through keras IO.

Scaling semantics (verified against the installed omlt 1.2.2 source,
``omlt/formulation.py::_setup_scaled_inputs_outputs``):

* When a scaling object is attached to the OMLT network, the OmltBlock
  variables ``block.inputs`` and ``block.outputs`` live in ORIGINAL
  (unscaled) space. OMLT adds constraints
  ``scaled_inputs == (inputs - offset_in) / factor_in`` and
  ``outputs == scaled_outputs * factor_out + offset_out``, and the
  neural-network constraints connect ``scaled_inputs`` to
  ``scaled_outputs``.
* ``omlt.neuralnet.NetworkDefinition(scaled_input_bounds=...)`` expects
  an int-keyed dict ``{i: (lb, ub)}`` of bounds in SCALED space; OMLT
  itself back-propagates them onto the unscaled ``block.inputs``.
* ``omlt.neuralnet.layer.DenseLayer`` evaluates
  ``np.dot(x, weights) + biases``, i.e. weights have shape
  ``(n_in, n_out)`` -- the same Keras kernel convention used by
  :class:`pwlnet.core.layers.DenseLayer`, so weights transfer verbatim.

All pyomo/omlt imports are lazy (inside functions).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pwlnet.utils.exceptions import FormulationError
from pwlnet.utils.logging import get_logger
from pwlnet.utils.typing import ArrayLike

__all__ = [
    "FORMULATIONS",
    "build_omlt_block",
    "build_omlt_formulation",
    "evaluate_block",
    "get_omlt_formulation",
    "to_omlt_network",
]

_logger = get_logger(__name__)

FORMULATIONS = ("relu_bigm", "tanh_exact")
"""Formulation names accepted by :func:`build_omlt_block`."""

#: pwlnet activation name -> OMLT activation name.
_OMLT_ACTIVATION = {"relu": "relu", "tanh": "tanh", "linear": "linear"}


def _make_scaling_object(network: Any) -> Any:
    """Build an ``omlt.OffsetScaling`` from the network's scalers.

    Returns ``None`` when the network has neither an input nor an
    output scaler (OMLT then treats inputs/outputs as already scaled).
    A missing scaler on one side is replaced by the identity
    (offset 0, factor 1).

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If any scaling factor is not strictly positive
        (``omlt.OffsetScaling`` rejects non-positive factors).
    """
    if network.input_scaler is None and network.output_scaler is None:
        return None

    from omlt import OffsetScaling

    if network.input_scaler is not None:
        offset_in = np.asarray(network.input_scaler.offset, dtype=float)
        factor_in = np.asarray(network.input_scaler.factor, dtype=float)
    else:
        offset_in = np.zeros(network.n_inputs)
        factor_in = np.ones(network.n_inputs)
    if network.output_scaler is not None:
        offset_out = np.asarray(network.output_scaler.offset, dtype=float)
        factor_out = np.asarray(network.output_scaler.factor, dtype=float)
    else:
        offset_out = np.zeros(network.n_outputs)
        factor_out = np.ones(network.n_outputs)

    if np.any(factor_in <= 0.0) or np.any(factor_out <= 0.0):
        raise FormulationError(
            "omlt.OffsetScaling requires strictly positive scaling factors; "
            "the network's OffsetScaler contains non-positive factor(s). "
            "Refit the scaler with positive factors before exporting to OMLT."
        )

    return OffsetScaling(
        offset_inputs=offset_in.tolist(),
        factor_inputs=factor_in.tolist(),
        offset_outputs=offset_out.tolist(),
        factor_outputs=factor_out.tolist(),
    )


def to_omlt_network(network: Any) -> Any:
    """Convert a pwlnet network to an ``omlt.neuralnet.NetworkDefinition``.

    The OMLT network is assembled layer by layer from the pwlnet
    layers (InputLayer followed by chained DenseLayers), with an
    ``OffsetScaling`` object built from the network's scalers and
    scaled input bounds taken from ``network.scaled_input_bounds()``.

    Parameters
    ----------
    network : pwlnet.core.network.NetworkDefinition
        The network to export. Must have ``input_bounds`` set
        (formulations require bounded inputs).

    Returns
    -------
    omlt.neuralnet.NetworkDefinition
        OMLT network with scaling object and scaled input bounds
        attached, ready to be passed to an OMLT formulation class.

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If ``network.input_bounds`` is ``None``, if an activation
        cannot be mapped, or if scaling factors are not strictly
        positive.
    """
    if network.input_bounds is None:
        raise FormulationError(
            "Cannot build an OMLT network without input bounds: set "
            "NetworkDefinition.input_bounds (original space) first."
        )
    network.validate()

    from omlt.neuralnet import NetworkDefinition as OmltNetworkDefinition
    from omlt.neuralnet.layer import DenseLayer as OmltDenseLayer
    from omlt.neuralnet.layer import InputLayer as OmltInputLayer

    scaling_object = _make_scaling_object(network)
    scaled_bounds = {
        i: (float(lb), float(ub))
        for i, (lb, ub) in enumerate(network.scaled_input_bounds())
    }

    omlt_net = OmltNetworkDefinition(
        scaling_object=scaling_object,
        scaled_input_bounds=scaled_bounds,
    )

    input_layer = OmltInputLayer([network.n_inputs])
    omlt_net.add_layer(input_layer)
    prev = input_layer
    for k, layer in enumerate(network.layers):
        try:
            omlt_activation = _OMLT_ACTIVATION[layer.activation]
        except KeyError as exc:
            raise FormulationError(
                f"Layer {k} activation '{layer.activation}' has no OMLT mapping. "
                f"Supported: {sorted(_OMLT_ACTIVATION)}."
            ) from exc
        dense = OmltDenseLayer(
            [layer.n_inputs],
            [layer.n_outputs],
            weights=np.asarray(layer.weights, dtype=float),
            biases=np.asarray(layer.biases, dtype=float),
            activation=omlt_activation,
        )
        omlt_net.add_layer(dense)
        omlt_net.add_edge(prev, dense)
        prev = dense

    _logger.debug(
        "Built OMLT network with %d dense layers (scaling=%s).",
        network.n_layers,
        "yes" if scaling_object is not None else "no",
    )
    return omlt_net


def _check_formulation_compatibility(network: Any, formulation: str) -> None:
    """Raise FormulationError if activations do not match the formulation."""
    hidden = network.activations[:-1]
    last = network.activations[-1]
    if formulation == "relu_bigm":
        required, allowed_last = "relu", {"relu", "linear"}
    else:  # tanh_exact
        required, allowed_last = "tanh", {"tanh", "linear"}
    bad = sorted({a for a in hidden if a != required})
    if bad:
        raise FormulationError(
            f"Formulation '{formulation}' requires all-{required} hidden "
            f"activations, but the network has hidden activations {bad}."
        )
    if last not in allowed_last:
        raise FormulationError(
            f"Formulation '{formulation}' supports an output layer activation in "
            f"{sorted(allowed_last)}, got '{last}'."
        )


def build_omlt_formulation(network: Any, formulation: str) -> Any:
    """Build an OMLT formulation object from a PWLNet network.

    Parameters
    ----------
    network : pwlnet.core.network.NetworkDefinition
        Network to encode. Must have ``input_bounds`` set.
    formulation : {"relu_bigm", "tanh_exact"}
        Formulation family.

    Returns
    -------
    object
        OMLT formulation object ready for ``OmltBlock.build_formulation``.
    """
    formulation_name = str(formulation).strip().lower()
    if formulation_name not in FORMULATIONS:
        raise FormulationError(
            f"Unknown formulation '{formulation}'. Supported: {list(FORMULATIONS)}."
        )
    _check_formulation_compatibility(network, formulation_name)
    omlt_net = to_omlt_network(network)
    if formulation_name == "relu_bigm":
        from pwlnet.formulations.omlt.relu_bigm import relu_bigm_formulation

        return relu_bigm_formulation(omlt_net)

    from pwlnet.formulations.omlt.tanh_exact import tanh_exact_formulation

    return tanh_exact_formulation(omlt_net)


get_omlt_formulation = build_omlt_formulation


def build_omlt_block(
    network: Any,
    formulation: str,
    model: Any | None = None,
    block_name: str = "neural_net",
) -> tuple[Any, Any]:
    """Attach an OmltBlock encoding the network to a Pyomo model.

    Parameters
    ----------
    network : pwlnet.core.network.NetworkDefinition
        The network to encode. Must have ``input_bounds`` set.
    formulation : {"relu_bigm", "tanh_exact"}
        ``"relu_bigm"`` builds ``omlt.neuralnet.ReluBigMFormulation``
        (MILP; requires all-relu hidden layers, relu/linear output).
        ``"tanh_exact"`` builds
        ``omlt.neuralnet.FullSpaceSmoothNNFormulation`` with exact
        ``y == tanh(x)`` constraints (smooth NLP; requires all-tanh
        hidden layers, tanh/linear output).
    model : pyomo.environ.ConcreteModel or None, optional
        Model to attach the block to. A fresh ``ConcreteModel`` is
        created when ``None``.
    block_name : str, optional
        Attribute name for the OmltBlock on the model.

    Returns
    -------
    tuple
        ``(model, block)`` where ``block`` is the constructed
        ``omlt.OmltBlock``. ``block.inputs`` / ``block.outputs`` are in
        ORIGINAL (unscaled) space whenever the network has scalers.

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        On unknown formulation name, incompatible activations, missing
        input bounds, or a ``block_name`` collision on ``model``.
    """
    formulation_name = str(formulation).strip().lower()
    if formulation_name not in FORMULATIONS:
        raise FormulationError(
            f"Unknown formulation '{formulation}'. Supported: {list(FORMULATIONS)}."
        )
    _check_formulation_compatibility(network, formulation_name)

    import pyomo.environ as pyo
    from omlt import OmltBlock

    formulation_obj = build_omlt_formulation(network, formulation_name)

    if model is None:
        model = pyo.ConcreteModel(name="pwlnet_omlt")
    if hasattr(model, block_name):
        raise FormulationError(
            f"Model already has a component named '{block_name}'; choose a "
            "different block_name."
        )
    block = OmltBlock()
    setattr(model, block_name, block)
    block.build_formulation(formulation_obj)
    _logger.debug(
        "Built OmltBlock '%s' with formulation '%s'.", block_name, formulation_name
    )
    return model, block


def _model_is_nonlinear(model: Any) -> bool:
    """True if any active constraint on the model is nonlinear."""
    import pyomo.environ as pyo

    for con in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        degree = con.body.polynomial_degree()
        if degree is None or degree > 1:
            return True
    return False


def _model_has_discrete_vars(model: Any) -> bool:
    """True if any variable on the model is binary or integer."""
    import pyomo.environ as pyo

    for var in model.component_data_objects(pyo.Var, descend_into=True):
        if var.is_binary() or var.is_integer():
            return True
    return False


def _check_solve_status(results: Any) -> None:
    """Raise FormulationError unless the solve terminated optimally.

    Handles both legacy Pyomo results (``results.solver.
    termination_condition``) and APPSI results
    (``results.termination_condition``).
    """
    import pyomo.environ as pyo

    try:
        if pyo.check_optimal_termination(results):
            return
    except (AttributeError, TypeError):
        pass
    condition = getattr(results, "termination_condition", None)
    if condition is not None and "optimal" in str(condition).lower():
        return
    raise FormulationError(
        f"OMLT block solve did not terminate optimally (results: {results})."
    )


def evaluate_block(
    model: Any,
    block: Any,
    x: ArrayLike,
    solver: Any | None = None,
) -> np.ndarray:
    """Evaluate an OmltBlock at a fixed input point by solving the model.

    Fixes ``block.inputs`` to ``x`` and solves. ``x`` must be given in
    ORIGINAL (unscaled) space: per the OMLT source
    (``omlt/formulation.py::_setup_scaled_inputs_outputs``),
    ``block.inputs``/``block.outputs`` are the unscaled variables when
    a scaling object is attached (OMLT adds the scaling constraints
    internally), and coincide with the scaled ones when it is not.

    Solver selection when ``solver`` is ``None``:

    * any nonlinear constraint present (tanh_exact) -> Ipopt via
      :func:`pwlnet.solvers.solver_config.get_nlp_solver`, which uses
      ``find_ipopt()`` and raises ``SolverNotAvailableError`` if no
      Ipopt executable is found;
    * otherwise (relu_bigm MILP, or pure LP) -> HiGHS via
      :func:`pwlnet.solvers.solver_config.get_milp_solver`.

    A constant dummy objective is added temporarily if the model has
    no active objective. The input variables are left fixed at ``x``
    after the call (repeated calls simply re-fix them).

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Model containing the block (as returned by
        :func:`build_omlt_block`).
    block : omlt.OmltBlock
        The block whose inputs are fixed and outputs read.
    x : array_like
        Input point in ORIGINAL space, shape ``(n_inputs,)``.
    solver : pyomo solver or None, optional
        Pre-constructed Pyomo solver to use instead of auto-selection.

    Returns
    -------
    numpy.ndarray
        Block outputs in ORIGINAL space, shape ``(n_outputs,)``.

    Raises
    ------
    pwlnet.utils.exceptions.FormulationError
        If ``x`` has the wrong length, the model mixes discrete
        variables with nonlinear constraints, or the solve does not
        terminate optimally.
    pwlnet.utils.exceptions.SolverNotAvailableError
        If the auto-selected solver (HiGHS or Ipopt) is unavailable.
    """
    import pyomo.environ as pyo

    from pwlnet.solvers.solver_config import get_milp_solver, get_nlp_solver

    x_arr = np.asarray(x, dtype=float).ravel()
    input_keys = list(block.inputs.keys())
    if x_arr.shape[0] != len(input_keys):
        raise FormulationError(
            f"x has {x_arr.shape[0]} entries but the block has "
            f"{len(input_keys)} inputs."
        )
    for key, value in zip(input_keys, x_arr):
        block.inputs[key].fix(float(value))

    if solver is None:
        nonlinear = _model_is_nonlinear(model)
        discrete = _model_has_discrete_vars(model)
        if nonlinear and discrete:
            raise FormulationError(
                "Model has both nonlinear constraints and discrete variables "
                "(MINLP); pass an appropriate solver explicitly."
            )
        solver = get_nlp_solver() if nonlinear else get_milp_solver()

    objective_added = False
    dummy_name = "_pwlnet_eval_dummy_objective"
    has_objective = any(
        True
        for _ in model.component_data_objects(
            pyo.Objective, active=True, descend_into=True
        )
    )
    if not has_objective:
        setattr(model, dummy_name, pyo.Objective(expr=0.0))
        objective_added = True
    try:
        results = solver.solve(model)
        _check_solve_status(results)
        outputs = np.array(
            [pyo.value(block.outputs[key]) for key in block.outputs.keys()],
            dtype=float,
        )
    finally:
        if objective_added:
            model.del_component(dummy_name)
    return outputs
