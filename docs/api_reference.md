# PWLNet API Reference

PWLNet is an optimization-aware neural-surrogate library for process systems
engineering. It trains compact dense and recurrent neural networks, stores them
in a framework-independent representation, prunes or compresses them for
smaller optimization models, and embeds them in Pyomo/OMLT, custom
piecewise-linear tanh formulations, SOSX branch-and-bound workflows, and
process-specific NMPC benchmarks.

This reference describes the public Python API and the notebooks in the current
repository. The package version in `pyproject.toml` and `pwlnet.__version__` is
`0.1.1`.

## Installation Extras

Pyomo and OMLT are included in the base installation, along with NumPy, SciPy,
and pandas. Add solver, training, and development support through extras.

Install the local checkout with the extras needed for the workflow:

```bash
pip install -e ".[highs,training,dev]"
```

Useful extras:

| Extra | Purpose |
|---|---|
| `optimization` or `highs` | Open-source HiGHS solver support |
| `gurobi` | Optional licensed Gurobi interfaces |
| `mosek` | Optional MOSEK API for conic, MIQP, and MISOCP paths |
| `training` | TensorFlow/Keras training support |
| `parallel` | Ray support for parallel SOSX training |
| `dev` | Matplotlib, pytest, notebook tooling, and Jupyter support |

Gurobi, MOSEK, and Ipopt are optional. The default open-source MILP/LP path is
HiGHS through Pyomo's `appsi_highs` interface.

## Package-Level API

Most common objects are re-exported from `pwlnet`:

```python
from pwlnet import (
    PWLNet,
    NetworkDefinition,
    DenseLayer,
    OffsetScaler,
    KerasTrainer,
    RNNTrainer,
    SosxTrainer,
    PhysicsInformedTrainer,
    PhysicsSpec,
    PhysicsLoss,
    WilliamsOttoPhysics,
    solve_sosx,
    train_test_split,
    train_val_test_split,
    regression_metrics,
)
```

### `PWLNet`

`PWLNet` is the user-facing facade around a trained `NetworkDefinition`. It is
intentionally thin: the wrapped network remains the source of truth, while
training, pruning, embedding, and reporting are delegated to specialist modules.

| Method | Description |
|---|---|
| `PWLNet.train(X, y, hidden_layers, activation="relu", validation_data=None, **trainer_options)` | Trains a dense Keras network (`activation` is `"relu"`, `"tanh"`, or `"pwl_tanh"`) and exports it to PWLNet's core representation. |
| `PWLNet.train_physics_informed(X, y, hidden_layers, activation="tanh", physics=None, physics_weights=None, **trainer_options)` | Trains with supervised data loss plus physics penalties. |
| `PWLNet.train_sosx(X, y, hidden_neurons, validation_data=None, initial_network=None, W2=None, incumbent=None, **trainer_options)` | Globally trains a fixed-readout, single-hidden-layer PWL-tanh network under MAE loss using SOSX. |
| `PWLNet.train_rnn(U, Y, hidden_units, validation_data=None, **trainer_options)` | Trains and returns a `RecurrentCell` for one-step ReLU RNN workflows. |
| `PWLNet.load(path)` | Loads a network JSON file and wraps it in `PWLNet`. |
| `save(path)` | Saves the wrapped network as JSON. |
| `predict(X)` | Evaluates the network in original units. |
| `summary()` | Returns a `ModelSummary`. |
| `bounds()` | Runs interval bound propagation and returns a `BoundsResult`. |
| `prune_report()` | Returns a `PruneReport` with size and pruning-readiness information. |
| `prune(method="dead_relu", fine_tune=False, X=None, y=None, validation_data=None, return_outcome=False, **kwargs)` | Returns a new pruned `PWLNet`; the source network is not mutated. |
| `compress(X=None, y=None, fine_tune=False, magnitude_fraction=None, neuron_fraction=None, return_outcome=False, **fine_tune_kwargs)` | Runs the recommended exact and optional approximate compression pipeline. |
| `to_pyomo(formulation="relu_bigm", backend="omlt", model=None, block_name="surrogate", **formulation_options)` | Builds a Pyomo/OMLT or custom Pyomo-PWL surrogate block. |

Supported facade export combinations:

| Backend | Formulation | Main use |
|---|---|---|
| `omlt` | `relu_bigm` | ReLU MILP embedding |
| `omlt` | `tanh_exact` | Smooth exact tanh NLP embedding |
| `pyomo_pwl` | `tanh_pwl_sos2` | Piecewise-linear tanh SOS2 embedding |
| `pyomo_pwl` | `tanh_pwl_bigm` | Piecewise-linear tanh big-M embedding |

Example:

```python
from pwlnet import PWLNet

net = PWLNet.train(
    X_train,
    y_train,
    hidden_layers=[16, 16],
    activation="tanh",
    epochs=500,
    validation_data=(X_val, y_val),
    scaling_method="standard",
    random_seed=7,
)

metrics_ready = net.predict(X_test)
model, block = net.to_pyomo(backend="pyomo_pwl", formulation="tanh_pwl_sos2")
```

## Core Network API

The `pwlnet.core` package contains the framework-independent dense-network
representation and utilities used by all optimization backends.

| Object | Description |
|---|---|
| `NetworkDefinition` | Ordered dense neural network with optional input/output scalers and original-unit input bounds. Supports prediction, JSON serialization, dimension metadata, and scaled-bound conversion. |
| `DenseLayer` | A fully connected layer `y = activation(x @ weights + biases)`. |
| `OffsetScaler` | Per-feature affine scaler with `x_scaled = (x - offset) / factor`. |
| `ActivationInfo` | Metadata record for supported activations. |
| `relu`, `tanh`, `identity` | NumPy activation functions. |
| `get_activation(name)` | Looks up activation metadata. |
| `BoundsResult` | Per-layer interval-bound propagation result. |
| `propagate_bounds(network)` | Propagates interval bounds through a network. |
| `ModelSummary` | Structural network summary. |
| `summarize(network)` | Builds a `ModelSummary`. |
| `estimate_formulation_size(network, formulation=...)` | Estimates formulation size for optimization planning. |

`NetworkDefinition` is the object to use when you need direct access to the
trained weights, activations, scalers, bounds, or JSON representation.

## Data And Metrics

| API | Description |
|---|---|
| `DataSplit` | Container for train, validation, and test partitions. |
| `train_val_test_split(X, y, ...)` | Splits one dataset into train/validation/test partitions. |
| `train_test_split(X, y, ...)` | Splits one dataset into train/test partitions. |
| `regression_metrics(y_true, y_pred)` | Computes aggregate MAE, MSE, RMSE, and R2. |
| `metrics_table({...})` | Builds a per-split metrics table for train/validation/test reporting. |

## Training API

### Dense Keras Training

| API | Description |
|---|---|
| `KerasTrainer` | Trains a dense feed-forward Keras model and exports it to `NetworkDefinition`. |
| `make_pwl_tanh_activation(breakpoints=5)` | Builds the PWL-tanh Keras activation used by `activation="pwl_tanh"`. |
| `make_early_stopping(...)` | Builds a Keras early-stopping callback. |
| `make_epoch_logger(...)` | Builds a lightweight logging callback. |
| `import_keras()` | Lazy Keras import helper. |
| `layer_seeds(base_seed, n_layers)` | Deterministic per-layer seed helper. |
| `get_initializer(...)` | Builds a seeded Keras initializer. |

#### PWL-Native Training (`activation="pwl_tanh"`)

A network trained with exact `tanh` and then embedded as PWL is read by
the solver through a function it was never fitted to. `"pwl_tanh"`
trains through the PWL tanh itself -- it is a sum of ReLUs, so it
backpropagates -- and the embedding then reproduces the trained model to
float round-off instead of approximating it. In notebook 09 this removed
the embedding mismatch that grew with the horizon (chain error
1.40 -> 0.19 bar at H=4, slack purchases to zero).

```python
net = PWLNet.train(
    X_train, y_train,
    hidden_layers=[16, 16],
    activation="pwl_tanh",
    breakpoints=5,          # MUST match the breakpoints used to embed
    epochs=500,
    validation_data=(X_val, y_val),
)
model, block = net.to_pyomo(
    formulation="tanh_pwl_sos2", backend="pyomo_pwl", breakpoints=5
)
```

| Detail | Behaviour |
|---|---|
| `breakpoints` | `3`, `5`, `7`, or an explicit vector spanning `[-4, 4]`. Only accepted with `activation="pwl_tanh"` -- passing it with another activation raises rather than being ignored. |
| Exported layers | Labelled `"tanh"` (`trainer.export_activation`), which is what makes the PWL embedding build its block; the core representation has no separate PWL activation. |
| `net.predict(X)` | Reads those layers with **exact** tanh, so it differs from what the solver sees. For the solver's reading use `evaluate_tanh_pwl_network(network, X, breakpoints=...)`. |
| `trainer.equivalence_error_` | Measured against that PWL reading (not `predict`), so it stays at ~1e-6 and still catches real export bugs. |

### Physics-Informed Training

| API | Description |
|---|---|
| `PhysicsInformedTrainer` | Trains dense networks with data loss plus physics loss. |
| `PhysicsLoss` | Base class for process-specific physics penalty modules. |
| `PhysicsSpec` | Name-based builder for affine physics-informed losses. |
| `GenericAffineResidualPhysics` | Adapter from affine residual definitions to trainer physics modules. |
| `register_physics(name, cls)` | Registers physics modules for string lookup. |
| `affine_residual_value(...)` | TensorFlow evaluation helper for affine residuals. |

### SOSX-MAE Training

| API | Description |
|---|---|
| `SosxTrainer` | Globally trains a bounded, fixed-readout, single-hidden-layer PWL-tanh network under MAE loss. |
| `SosxTrainingModel` | Built Pyomo SOSX-MAE training model and handles. |
| `build_sosx_mae_training_model(...)` | Constructs the fixed-readout SOSX-MAE training model. |
| `SosxTrainingRelaxation` | Reusable LP relaxation for SOSX-MAE training. |
| `WeightBounds` | Symmetric bounds for trainable SOSX parameters. |
| `resolve_weight_bounds(...)` | Resolves effective parameter bounds. |
| `bounds_covering_network(network)` | Builds bounds wide enough to contain a warm-start network. |
| `track_training_variables(...)` | Builds a brancher tracking map for trainable variables. |
| `network_from_tracked_values(...)` | Rebuilds a network from a brancher incumbent. |
| `training_incumbent_from_network(...)` | Converts a one-hidden-layer network into a SOSX warm start. |
| `predict_pwl_tanh(...)` | Evaluates the shared PWL-tanh approximation in NumPy. |
| `pwl_training_mae(...)` | Computes MAE under the PWL-tanh training approximation. |
| `ParallelSosxBrancher` | Ray-backed parallel SOSX brancher. |
| `SosxTrainingSpec` | Worker rebuild specification for parallel SOSX training. |
| `default_worker_count()` | Conservative Ray worker-count helper. |

## Pruning And Compression

| API | Description |
|---|---|
| `PruneOutcome` | Result object for one pruning operation. |
| `PruneReport` | Compact report of network size and pruning opportunities. |
| `build_prune_report(network)` | Builds a pruning/readiness report. |
| `prune_network(network, method, **kwargs)` | Dispatches to a named pruning method. |
| `prune_dead_relus(network)` | Removes ReLU neurons that are provably always zero over the input box. |
| `simplify_always_active_relus(network)` | Linearizes ReLU units that are always active. |
| `prune_relus(network)` | Applies exact dead-ReLU removal and always-active simplification. |
| `prune_saturated_tanh(network)` | Folds saturated tanh neurons to constants. |
| `prune_weights_magnitude(network, ...)` | Magnitude-prunes weights and optionally cleans up trivial neurons. |
| `neuron_importance(network, ...)` | Computes per-neuron importance scores. |
| `prune_neurons(network, ...)` | Removes the least-important hidden neurons. |
| `group_lasso_penalty(...)` | Computes group-lasso penalty values. |
| `make_group_lasso_regularizer(...)` | Builds a Keras group-lasso regularizer. |
| `prune_by_group_norm(network, ...)` | Removes neurons below a group-norm threshold. |
| `obd_saliencies(network, X, y, ...)` | Computes Optimal Brain Damage saliencies. |
| `optimal_brain_damage(network, X, y, ...)` | Prunes lowest-saliency weights. |
| `fine_tune(network, X, y, ...)` | Fine-tunes a pruned network with Keras. |
| `compare_networks(before, after, ...)` | Builds a before/after structural and formulation comparison. |

Supported `PWLNet.prune(...)` method names include `dead_relu`,
`always_active`, `always_active_relu`, `relus`, `relu`, `saturated_tanh`,
`weight_magnitude`, `magnitude`, `neuron`, `neuron_magnitude`, `group_lasso`,
and `obd`.

## Formulation And Embedding API

### OMLT

| API | Description |
|---|---|
| `to_omlt_network(network)` | Converts a PWLNet network to an OMLT network definition. |
| `build_omlt_formulation(network, formulation=...)` | Builds an OMLT formulation object. |
| `build_omlt_block(network, formulation=..., model=None, block_name="surrogate")` | Attaches an OMLT block to a Pyomo model. |
| `evaluate_block(model, block, X, ...)` | Evaluates an OMLT block at fixed inputs by solving the model. |
| `relu_bigm_formulation(network)` | Builds an OMLT ReLU big-M formulation. |
| `tanh_exact_formulation(network)` | Builds an OMLT smooth tanh formulation. |

### Custom Pyomo PWL Tanh

| API | Description |
|---|---|
| `validate_breakpoints(values)` | Validates a strictly increasing breakpoint vector. |
| `get_tanh_breakpoints(breakpoints)` | Resolves standard or custom tanh breakpoints. |
| `PiecewiseLinearData` | Slopes, intercepts, breakpoints, and function values for a PWL approximation. |
| `build_piecewise_linear_data(x, y)` | Builds segment data from breakpoints and values. |
| `extend_piecewise_linear_data(data, lower, upper)` | Extends a PWL approximation with flat outer segments. |
| `build_tanh_pwl_data(breakpoints)` | Builds tanh-specific PWL data. |
| `build_tanh_pwl_sos2_block(model, network, ...)` | Attaches an SOS2 PWL-tanh block to a Pyomo model. |
| `build_tanh_pwl_sos2_model(network, ...)` | Creates a model containing one SOS2 PWL-tanh block. |
| `build_tanh_pwl_bigm_block(model, network, ...)` | Attaches a big-M PWL-tanh block to a Pyomo model. |
| `build_tanh_pwl_bigm_model(network, ...)` | Creates a model containing one big-M PWL-tanh block. |
| `evaluate_tanh_pwl_network(network, X, ...)` | Evaluates the PWL tanh approximation in NumPy. |
| `add_quadratic_epigraph(model, errors, ...)` | Adds a conic quadratic epigraph for squared-error objectives. |

### Generic Embedding Helpers

| API | Description |
|---|---|
| `resolve_omlt_network(network)` | Resolves an OMLT network definition. |
| `build_surrogate_block(model, network, ...)` | Attaches a surrogate block to an existing Pyomo model. |

## SOSX Optimization API

The SOSX solver handles models where tanh activations are represented with
piecewise-linear convex-combination variables and SOS2 validity is enforced by
branching rather than by binary variables.

| API | Description |
|---|---|
| `solve_sosx(model_or_network, ...)` | Solves a SOSX model or the built-in Williams-Otto surrogate workflow. |
| `SosxSurrogateResult` | Williams-Otto convenience result returned by `solve_sosx(network, ...)`. |
| `SosxResult` | Final branch-and-bound report for model-based SOSX solves. |
| `NodeSolution` | Result of solving one node relaxation. |
| `SosxBrancher` | Branch-and-bound engine over continuous SOS2 weights. |
| `SosxRelaxation` | Reusable SOS2-dropped relaxation of a tanh surrogate model. |
| `discover_sos2_blocks(model)` | Finds blocks that carry SOS2 tanh activations. |
| `find_violations(...)` | Finds non-SOS2 convex-combination supports. |
| `find_first_violation(...)` | Returns the first SOS2 violation. |
| `support_spread(...)` | Measures support width in one SOS set. |
| `select_violation(...)` | Selects the SOS set to branch on. |
| `split_window(...)` | Splits a violated SOS window into children. |
| `Node` | One open branch-and-bound node. |
| `Frontier` | Open-node container with best-bound or depth-first selection. |
| `Incumbent` | Best SOS2-feasible node found so far. |
| `BranchLog` / `BranchRecord` | Optional branch trace logging. |
| `can_fathom_by_bound(...)` | Bound-based subtree pruning check. |
| `is_improving(...)` | Incumbent improvement check. |

Important `solve_sosx` options include `solver`, `solver_options`,
`node_selection`, `branching_rule`, `max_nodes`, `time_limit`, `absolute_gap`,
`relative_gap`, `optimality_tolerance`, and `violation_tolerance`.

## Objectives

| API | Description |
|---|---|
| `DataLossResult` | Handles for a data-loss term added to a Pyomo model. |
| `add_mae_loss(model, predictions, targets, ...)` | Adds mean-absolute-error epigraph variables and constraints. |
| `add_mse_qp_loss(model, predictions, targets, ...)` | Adds direct quadratic MSE loss. |
| `add_mse_socp_loss(model, predictions, targets, ...)` | Adds conic epigraph MSE loss. |
| `attach_data_loss(model, predictions, targets, kind=..., objective=True)` | Adds data loss and optionally creates an objective. |

## Physics Residual API

| API | Description |
|---|---|
| `PhysicsResidual` | Abstract physical-consistency residual. |
| `AffineResidual` | Affine residual `r = x @ W_in.T + y @ W_out.T + b`. |
| `linear_residual(...)` | Builds a sparse affine residual. |
| `sum_to_one_residual(...)` | Mass-fraction closure residual. |
| `mass_balance_residual(...)` | Single-row linear mass balance residual. |
| `conservation_residual(...)` | Multi-row linear conservation residual. |
| `upper_bound_residual(...)` | Upper-bound violation residual. |
| `lower_bound_residual(...)` | Lower-bound violation residual. |
| `nonnegativity_residual(...)` | Nonnegativity violation residual. |
| `box_residual(...)` | Two-sided bound residual. |
| `MonotonicityConstraint` | Finite-difference monotonicity requirement. |
| `finite_difference_violation(...)` | Monotonicity violation by finite differences. |
| `available_residuals()` | Lists registered residual names. |
| `build_residual(name, **kwargs)` | Builds a registered residual. |

## Recurrent And MPC API

| API | Description |
|---|---|
| `RNNTrainer` | Trains a one-step ReLU SimpleRNN and exports a `RecurrentCell`. |
| `RecurrentCell` | Self-contained recurrent cell with NumPy rollout, pruning, bounds, and MPC builders. |
| `info_is_pwl(activation)` | Checks whether an activation is piecewise linear. |
| `RecurrentBounds` | Per-step bounds from recurrent interval propagation. |
| `propagate_recurrent_bounds(cell, ...)` | Propagates hidden/output bounds over an unrolled horizon. |
| `build_unrolled_mpc_model(cell, ...)` | Builds a ReLU-MIP unrolled MPC model. |
| `solve_unrolled_mpc(cell, ...)` | Builds and solves one MPC move. |

## Solver Configuration

| API | Description |
|---|---|
| `get_pyomo_solver(name, options=None)` | Constructs a configured Pyomo solver. |
| `get_milp_solver(name=None, options=None)` | Constructs a MILP/LP solver, defaulting to an open-source path when possible. |
| `get_conic_solver(name=None, options=None)` | Constructs a conic solver. |
| `get_nlp_solver(name=None, options=None)` | Constructs an NLP solver. |
| `find_ipopt()` | Locates an Ipopt executable if available. |
| `mosek_license_available()` | Checks whether MOSEK is importable and can check out a license. |

## Process Modules

### Williams-Otto

Import from `pwlnet.processes.williams_otto`.

| API | Description |
|---|---|
| `WilliamsOttoParameters` | Canonical benchmark constants, bounds, prices, and operating limits. |
| `rate_constants(tr, params)` | Arrhenius rate constants. |
| `residuals(...)` | Six steady-state component-balance residuals. |
| `profit(...)` | Economic objective at an operating point. |
| `WilliamsOttoSteadyState` | Solved steady-state record. |
| `solve_steady_state(FB, TR, ...)` | Solves the steady-state equations. |
| `simulate_points(X, ...)` | Simulates many `(FB, TR)` points. |
| `generate_dataset(n_samples, ...)` | Generates supervised training data. |
| `WilliamsOttoPhysics` | Closure and operating-bound penalties for physics-informed training. |
| `SurrogateOptimizationResult` | Report from embedded surrogate optimization. |
| `build_exact_model(...)` | Builds the exact steady-state Pyomo model. |
| `build_surrogate_optimization_model(network, ...)` | Builds a Williams-Otto economic model with an embedded network. |
| `solve_surrogate_optimization_model(network, ...)` | Builds and solves the embedded surrogate model. |

### Hydroformylation

Import from `pwlnet.processes.hydroformylation`.

| API | Description |
|---|---|
| `HydroformylationParameters` | Kinetic and reactor parameters. |
| `dissolved_gas_equilibrium_mol_l(...)` | Henry-law dissolved gas concentration. |
| `pressure_backoff(...)` | Pressure safety backoff for NMPC. |
| `sample_parameters(...)` | Samples uncertain kinetic/transfer parameters. |
| `arrhenius_table6(...)` | Temperature-adjusts kinetic constants. |
| `equilibrium_constant(...)` | Computes the equilibrium constant from table coefficients. |
| `rhs(t, state, control, params)` | Semibatch plant ODE right-hand side. |
| `integrate_interval(...)` | Integrates one constant-control interval. |
| `step_state(...)` | Advances the plant by one interval. |
| `simulate_policy(...)` | Simulates a coarse policy on the true plant. |
| `simulate_sample_policy(...)` | Simulates a 15-second policy on the true plant. |
| `delayed_measurements(...)` | Produces sampled and delayed measurements. |
| `random_control_profiles(...)` | Generates coarse random profiles around the optimum. |
| `make_transition_dataset(...)` | Generates one-step transition data for 5-minute surrogates. |
| `random_sample_control_profiles(...)` | Generates 15-second random profiles. |
| `make_sample_transition_dataset(...)` | Generates one-step transition data for 15-second surrogates. |
| `reachable_input_bounds(...)` | Builds tight embedding bounds from transition data. |
| `tighten_network_bounds(network, bounds)` | Returns a network copy with restricted input bounds. |
| `solve_receding_move(...)` | Solves one coarse one-block PWL-tanh MILP control move. |
| `run_receding_horizon_nmpc(...)` | Runs coarse closed-loop NMPC. |
| `latest_delayed_measurement_time(...)` | Finds the newest available delayed sample time. |
| `estimate_state_from_delayed_measurement(...)` | Propagates delayed measurements to the current decision time. |
| `solve_15s_gurobi_milp_move(...)` | Solves one chained 15-second PWL-tanh MILP move. |
| `run_15s_gurobi_milp_nmpc(...)` | Runs 15-second closed-loop PWL-tanh MILP NMPC. |
| `solve_open_loop_optimal_control(...)` | Direct single-shooting optimal control on the true plant. |
| `interpolate_control_profile(...)` | Interpolates coarse controls to a target grid. |
| `optimal_control_profiles(...)` | Returns sampled model-based optimal profiles. |

### Van de Vusse

Import from `pwlnet.processes.van_de_vusse`.

| API | Description |
|---|---|
| `VanDeVusseParameters` | Isothermal CSTR parameter set. |
| `rhs(...)` | Dynamic concentration derivatives. |
| `steady_state(D, ...)` | Analytic steady state. |
| `steady_state_residuals(...)` | Steady-state balance residuals. |
| `product_concentration(...)` | Desired-product concentration. |
| `VanDeVusseSteadyState` | Solved steady-state record. |
| `solve_steady_state(...)` | Evaluates steady state at a dilution rate. |
| `simulate_points(...)` | Simulates many steady-state points. |
| `generate_dataset(...)` | Generates steady-state supervised data. |
| `find_max_product_dilution(...)` | Finds the steady-state dilution rate that maximizes `CB`. |
| `step_state(...)` | Advances the dynamic plant over one interval. |
| `simulate_policy(...)` | Simulates a piecewise-constant dilution policy. |
| `simulate_step_response(...)` | Simulates a dilution-rate step. |
| `make_transition_dataset(...)` | Generates one-step NMPC transition data. |
| `build_multiple_shooting_model(...)` | Builds a chained transition-surrogate control model. |
| `solve_multiple_shooting_move(...)` | Solves one surrogate MPC move. |
| `run_receding_horizon_nmpc(...)` | Runs closed-loop Van de Vusse NMPC. |
| `solve_direct_shooting_optimal_control(...)` | Solves direct-shooting open-loop control. |
| `direct_shooting_random_starts(...)` | Runs multistart direct shooting. |
| `save_optimal_control_solution(...)` | Saves an optimal-control `.npz` file. |
| `SahlodinVanDeVusseParameters` | Sahlodin et al. reactor parameters. |
| `reaction_rates(...)`, `outlet_flow(...)`, `heat_release(...)` | Sahlodin-model helper equations. |

## Utilities And Errors

| API | Description |
|---|---|
| `configure_logging(level=...)` | Configures the `pwlnet` root logger. |
| `get_logger(name)` | Returns a logger in the `pwlnet` namespace. |
| `as_2d_array(x, ...)` | Coerces input to a 2D float array. |
| `check_shape(array, expected, ...)` | Validates one array shape. |
| `check_shapes({...})` | Validates multiple shapes. |
| `PWLNetError` | Base package exception. |
| `NetworkValidationError` | Invalid network structure. |
| `FormulationError` | Optimization formulation construction failure. |
| `SolverNotAvailableError` | Missing or unusable solver. |
| `TrainingError` | Training failure or unusable trained model. |

## Notebook Reference And Results

The notebooks are executable project walkthroughs. The short result notes below
summarize the current saved outputs in the notebooks where available.

| Notebook | Description | Short result summary |
|---|---|---|
| `notebooks/01_williams_otto_relu_omlt_baseline.ipynb` | ReLU/OMLT Williams-Otto baseline: generates a 150-point dataset, trains a compact ReLU surrogate, checks Keras/core/OMLT consistency, embeds the surrogate in Pyomo, visualizes the profit and feasibility landscape, and compares constrained vs unconstrained operating constraints. | Saved run shows a 90/30/30 split, a compact ReLU network, parity/profit plots, and an embedded surrogate optimization section. Prior validated output recorded OMLT/core agreement at about `5.55e-17`, a prune report with 8 hidden ReLU neurons, and a 56-variable/72-constraint embedded surrogate model. |
| `notebooks/02_williams_otto_tanh_omlt_benchmark.ipynb` | Tanh surrogate benchmark using exact smooth OMLT embedding. It compares the learned surrogate landscape and embedded optimum against the exact Williams-Otto benchmark. | Saved run reports test MAE about `8.80e-4`, test RMSE about `1.48e-3`, and test R2 about `0.9984` on the 30-point test split, with optimization comparison plots. |
| `notebooks/03_williams_otto_tanh_pwl_milp.ipynb` | Manual PWL-tanh MILP benchmark for Williams-Otto. It studies tanh approximation quality, trains a smooth tanh surrogate, compares exact NLP, SOS2, and big-M PWL embeddings, and separates surrogate feasibility from exact-simulator feasibility. | Saved run found Ipopt and MOSEK available. PWL test MAE improved from about `0.00340` with 3 pieces to `0.00242` with 7 pieces, while max test deviation from the smooth surrogate stayed around `0.010-0.012`. |
| `notebooks/04_hydroformylation_nmpc_pwl_control.ipynb` | Hydroformylation greedy NMPC using state-space transition surrogates and Gurobi. It builds transition data, trains source-code surrogates, embeds them, and compares greedy control over 10-minute and 70-minute horizons. | Saved run shows Gurobi available, a 15-second decision step, known optimal terminal `nC13al = 0.5587`, and 4,480 generated state-space transitions split into 3,136/672/672 train/validation/test rows. A later edit changed the short multiple-shooting comparison to configurable chunked solves using `SHORT_CONTROL_HORIZON_MIN` and `MS_CHUNK_HORIZON_MIN`. |
| `notebooks/05_williams_otto_relu_pruning.ipynb` | Phase-3 ReLU pruning and optimization-aware compression. It trains a compact surrogate, measures embedded optimization size and solve behavior, applies exact ReLU simplification, then magnitude/neuron pruning with optional fine-tuning. | Saved run reports baseline test MAE about `7.61e-4`, 20 hidden neurons, 205 nonzero parameters, 20 binaries, and 106 constraints before pruning. Exact-only compression preserved test predictions with max change `0.00e+00`. |
| `notebooks/06_williams_otto_physics_informed_training.ipynb` | Phase-4 Williams-Otto physics-informed training. It diagnoses mass-balance, closure, and operating-bound residuals, trains a data-only baseline and physics-informed variants, and compares accuracy, physics residuals, and downstream optimization behavior. | Saved run records exact-label balance residuals near numerical precision, closure max error near `2.22e-16`, weighted constraint outputs (`XA=8`, `XG=8`), and a 32-hidden-neuron tanh model with Keras/core prediction difference about `1.77e-8`. |
| `notebooks/07_hydroformylation_physics_informed_training.ipynb` | Hydroformylation physics-informed transition training. It builds transition data, computes pressure/bounds/monotonicity diagnostics, trains data-only and physics-informed models, and compares prediction scatter and residuals. | Saved run uses 1,792/384/384 train/validation/test rows, exact-transition physics scores of zero for the diagnostic residuals, and a 32-hidden-neuron tanh model with Keras/core prediction difference about `7.45e-7`. |
| `notebooks/08_williams_otto_sosx_branch_and_lp.ipynb` | Williams-Otto comparison of smooth, physics-informed, and SOSX-refined tanh surrogates. It compares prediction pipelines, validation losses, CPU training time, and constrained Williams-Otto solve performance. | Saved run uses 120 samples, 6 hidden neurons, 5 breakpoints, and `appsi_highs`. It reports smooth fit around 22.3 s, PI fit around 3.9 s, and SOSX refinement around 0.7 s with optimal status, 2 nodes, and zero reported gap. |
| `notebooks/09_hydroformylation_sosx_shrinking_horizon.ipynb` | Hydroformylation 15-second NMPC comparison of SOSX and faster smooth/MILP methods over a shrinking horizon. It trains several surrogates, builds embedded move solvers, and compares solution quality, time, and variable counts. | Saved run finds Ipopt, uses a 15-second decision step, a 5-minute closed-loop window, 20 decisions, 3 breakpoints, hidden width 16, 150 epochs, and 7,680/1,920 train/test transition rows. Plain nets trained in about 17 s in the saved output. |
| `notebooks/10_williams_otto_sosx_input_noise_stress_test.ipynb` | Williams-Otto SOSX robustness test under noisy training inputs. It trains clean and noisy SOSX models, compares clean/noisy test metrics, repeated noisy-test robustness, CPU training time, and solve performance with piecewise McCormick economics. | Saved run uses 5% input-range noise, 120 samples, 6 hidden neurons, and 5 breakpoints. Clean-input SOSX training took about 19.9 s and noisy-input SOSX about 9.9 s; both saved runs report optimal status, 2 nodes, and zero gap. |
| `notebooks/11_van_de_vusse_nmpc_pwl_control.ipynb` | Van de Vusse NMPC notebook using terminal-horizon output surrogates and receding control. It prepares a refined 6-second reference path, builds terminal-horizon `CB` datasets, trains surrogates, evaluates terminal prediction accuracy, and runs receding terminal-horizon control. | The notebook currently has no saved executed outputs, so the documented result is the workflow design rather than a cached numeric benchmark. |
| `notebooks/12_van_de_vusse_relu_rnn_mpc.ipynb` | Physics-informed ReLU-RNN surrogate for Van de Vusse, pruned and embedded in a ReLU-MIP MPC model. It builds sequence data, applies residual and high-bound physics penalties, trains a ReLU RNN, prunes it, tightens bounds across the horizon, and solves one MPC move. | Saved run uses 120 training sequences of length 40 with one control and four outputs. Validation MAE per output is about `CA=0.0648`, `CB=0.0194`, `CC=0.1012`, and `CD=0.0289` for the 32-unit model. |

## Recommended Reading Path

For a first API pass, read the notebooks in this order:

1. `notebooks/01_williams_otto_relu_omlt_baseline.ipynb` for the core ReLU/OMLT workflow.
2. `notebooks/03_williams_otto_tanh_pwl_milp.ipynb` for tanh PWL formulations and feasibility interpretation.
3. `notebooks/05_williams_otto_relu_pruning.ipynb` for compression and optimization-aware pruning.
4. `notebooks/08_williams_otto_sosx_branch_and_lp.ipynb` for SOSX training and branch-and-LP optimization.
5. `notebooks/04_hydroformylation_nmpc_pwl_control.ipynb` and `notebooks/09_hydroformylation_sosx_shrinking_horizon.ipynb` for dynamic NMPC workflows.
6. `notebooks/12_van_de_vusse_relu_rnn_mpc.ipynb` for recurrent surrogate embedding.

## Current Scope Notes

PWLNet is still an alpha research codebase. The core package already exposes a
coherent `src/` layout, public facade API, optional solver extras, benchmark
process modules, and tests. Some workflows remain notebook-driven and rely on
optional solvers or local licenses, especially Gurobi, MOSEK, and Ipopt. When
reporting constrained surrogate results, distinguish feasibility of the
embedded surrogate optimization model from feasibility under the exact process
simulator.
