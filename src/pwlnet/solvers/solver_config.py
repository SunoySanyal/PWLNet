"""Solver discovery and construction helpers.

HiGHS (via Pyomo's ``appsi_highs`` interface) is the default open-source
MILP/LP solver. Gurobi interfaces are optional and require a licensed
installation. Ipopt is OPTIONAL: :func:`find_ipopt` searches the
``PWLNET_IPOPT`` environment variable, the system ``PATH``, and the
project-local ``.tools/ipopt/**/bin/ipopt(.exe)`` download location.
Pyomo is imported lazily inside the functions so that importing this
module never requires pyomo.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from pwlnet.config import (
    DEFAULT_CONIC_SOLVER,
    DEFAULT_MILP_SOLVER,
    DEFAULT_NLP_SOLVER,
    IPOPT_ENV_VAR,
)
from pwlnet.utils.exceptions import SolverNotAvailableError
from pwlnet.utils.logging import get_logger

__all__ = [
    "find_ipopt",
    "get_conic_solver",
    "get_milp_solver",
    "get_nlp_solver",
    "get_pyomo_solver",
    "mosek_license_available",
]

_logger = get_logger(__name__)


def _candidate_tool_roots() -> list[Path]:
    """Directories that may contain a project-local ``.tools/ipopt`` tree."""
    roots: list[Path] = []
    # Project root when running from a source checkout:
    # <root>/src/pwlnet/solvers/solver_config.py -> parents[3] == <root>
    try:
        roots.append(Path(__file__).resolve().parents[3])
    except IndexError:  # pragma: no cover - very shallow install layouts
        pass
    roots.append(Path.cwd())
    return roots


def find_ipopt() -> str | None:
    """Locate an Ipopt executable, if one is available.

    Search order:

    1. The ``PWLNET_IPOPT`` environment variable (must point at an
       existing file).
    2. ``shutil.which("ipopt")`` on the system ``PATH``.
    3. A project-local download under ``<project>/.tools/ipopt/**/bin/
       ipopt(.exe)`` (also checked relative to the current working
       directory).

    Returns
    -------
    str or None
        Absolute path to the Ipopt executable, or ``None`` if not
        found. Ipopt is optional everywhere in PWLNet; callers should
        degrade gracefully when this returns ``None``.
    """
    env_path = os.environ.get(IPOPT_ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return str(p.resolve())
        _logger.warning(
            "%s is set to '%s' but no file exists there; continuing search.",
            IPOPT_ENV_VAR,
            env_path,
        )

    which = shutil.which("ipopt")
    if which:
        return str(Path(which).resolve())

    for root in _candidate_tool_roots():
        tool_dir = root / ".tools" / "ipopt"
        if not tool_dir.is_dir():
            continue
        for pattern in ("**/bin/ipopt.exe", "**/bin/ipopt"):
            for hit in sorted(tool_dir.glob(pattern)):
                if hit.is_file():
                    return str(hit.resolve())

    return None


def _apply_solver_options(solver: Any, options: dict[str, Any] | None) -> None:
    """Apply options to classic Pyomo and APPSI solver wrappers."""
    if not options:
        return
    solver_options = getattr(solver, "options", None)
    for key, value in options.items():
        applied = False
        if solver_options is not None:
            try:
                solver_options[key] = value
                applied = True
            except Exception:  # pragma: no cover - exotic solver config object
                _logger.debug(
                    "Could not set solver option %s on solver.options.", key
                )
        if not applied and hasattr(solver, "config"):
            try:
                solver.config[key] = value
                applied = True
            except Exception:  # pragma: no cover - exotic solver config object
                _logger.debug(
                    "Could not set solver option %s on solver.config.", key
                )
        if not applied:
            raise SolverNotAvailableError(
                f"Solver {solver!r} does not expose a usable options/config "
                f"mapping for option '{key}'."
            )


def _check_available(solver: Any, solver_name: str) -> None:
    """Raise a clear package error if a Pyomo solver is unavailable."""
    try:
        available = bool(solver.available(exception_flag=False))
    except TypeError:
        available = bool(solver.available())
    if solver is None or not available:
        hint = ""
        if solver_name in {"appsi_highs", "highs"}:
            hint = " Install the HiGHS extra with 'pip install pwlnet[highs]'."
        elif "gurobi" in solver_name:
            hint = (
                " Gurobi is optional and requires 'gurobipy' plus a valid "
                "license; install with 'pip install pwlnet[gurobi]'."
            )
        elif solver_name == "ipopt":
            hint = (
                " Ipopt is optional; set PWLNET_IPOPT, add ipopt to PATH, "
                "or place a build under <project>/.tools/ipopt/**/bin/."
            )
        elif "mosek" in solver_name:
            hint = (
                " MOSEK is optional and requires 'pip install mosek' plus a "
                "license file in ~/mosek/mosek.lic (or pointed at by the "
                "MOSEKLM_LICENSE_FILE environment variable)."
            )
        raise SolverNotAvailableError(
            f"Pyomo solver '{solver_name}' is not available.{hint}"
        )


def get_pyomo_solver(
    name: str | None = None,
    options: dict[str, Any] | None = None,
    executable: str | os.PathLike[str] | None = None,
) -> Any:
    """Construct a configured Pyomo solver by factory name.

    Parameters
    ----------
    name : str or None, optional
        Pyomo solver factory name. ``None`` uses
        :data:`pwlnet.config.DEFAULT_MILP_SOLVER` (``"appsi_highs"``).
        Common names include ``"appsi_highs"``, ``"highs"``,
        ``"gurobi"``, ``"gurobi_direct"``, ``"gurobi_persistent"``,
        ``"appsi_gurobi"``, and ``"ipopt"``.
    options : dict or None, optional
        Solver options applied to ``solver.options`` when available,
        with a fallback to ``solver.config`` for APPSI-style wrappers.
    executable : str or os.PathLike or None, optional
        Explicit executable path for shell-based solvers. For ``ipopt``,
        ``None`` preserves the project discovery behavior in
        :func:`find_ipopt`.

    Returns
    -------
    object
        A Pyomo solver object ready for ``solve(model)``.

    Raises
    ------
    pwlnet.utils.exceptions.SolverNotAvailableError
        If Pyomo is missing, a requested executable cannot be found, or
        the solver factory reports unavailable.
    """
    solver_name = (name or DEFAULT_MILP_SOLVER).strip()
    try:
        import pyomo.environ  # noqa: F401, PLC0415 -- registers solver plugins
        from pyomo.opt import SolverFactory
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SolverNotAvailableError(
            "pyomo is not installed; install it with 'pip install pwlnet[optimization]'."
        ) from exc

    factory_kwargs: dict[str, Any] = {}
    if solver_name == "ipopt":
        if executable is None:
            executable = find_ipopt()
        if executable is None:
            raise SolverNotAvailableError(
                "Ipopt executable not found. Set the PWLNET_IPOPT environment "
                "variable to the full path of ipopt(.exe), add ipopt to PATH, "
                "or place a build under <project>/.tools/ipopt/**/bin/. "
                "Ipopt is optional: NLP-based features should be skipped "
                "gracefully when it is unavailable."
            )
        factory_kwargs["executable"] = str(executable)
    elif executable is not None:
        factory_kwargs["executable"] = str(executable)

    solver = SolverFactory(solver_name, **factory_kwargs)
    _check_available(solver, solver_name)
    _apply_solver_options(solver, options)
    _logger.debug("Constructed Pyomo solver '%s'.", solver_name)
    return solver


def get_milp_solver(name: str | None = None, options: dict[str, Any] | None = None) -> Any:
    """Construct a configured Pyomo MILP/LP solver.

    Parameters
    ----------
    name : str or None, optional
        Pyomo solver factory name. Defaults to
        :data:`pwlnet.config.DEFAULT_MILP_SOLVER` (``"appsi_highs"``).
    options : dict or None, optional
        Solver options applied to the returned solver object.

    Returns
    -------
    object
        A Pyomo solver object ready for ``solve(model)``.

    Raises
    ------
    pwlnet.utils.exceptions.SolverNotAvailableError
        If pyomo is not installed or the requested solver is not
        available.
    """
    return get_pyomo_solver(name=name or DEFAULT_MILP_SOLVER, options=options)


def mosek_license_available() -> bool:
    """Check whether MOSEK is importable and can check out a license.

    Performs an actual license checkout through the MOSEK Optimizer API
    (the Pyomo factory's ``available()`` only verifies that the
    ``mosek`` package imports; license problems would otherwise surface
    only at solve time).

    Returns
    -------
    bool
        ``True`` when the ``mosek`` package imports and a PTON
        (optimizer) license can be checked out.
    """
    try:
        import mosek
    except ImportError:
        return False
    try:
        with mosek.Env() as env:
            env.checkoutlicense(mosek.feature.pton)
            env.checkinlicense(mosek.feature.pton)
    except mosek.Error as exc:
        _logger.debug("MOSEK license checkout failed: %s", exc)
        return False
    return True


def get_conic_solver(
    name: str | None = None, options: dict[str, Any] | None = None
) -> Any:
    """Construct a configured Pyomo conic (SOCP/MIQP/MISOCP) solver.

    Parameters
    ----------
    name : str or None, optional
        Pyomo solver factory name. Defaults to
        :data:`pwlnet.config.DEFAULT_CONIC_SOLVER` (``"mosek_direct"``).
        Gurobi names also work for quadratic/conic problem classes.
    options : dict or None, optional
        Solver options applied to the returned solver object.

    Returns
    -------
    object
        A Pyomo solver object ready for ``solve(model)``.

    Raises
    ------
    pwlnet.utils.exceptions.SolverNotAvailableError
        If pyomo is not installed, the requested solver is unavailable,
        or (for MOSEK) no license can be checked out. MOSEK is optional
        in PWLNet; callers that can fall back (or skip) should catch
        this exception.
    """
    solver_name = (name or DEFAULT_CONIC_SOLVER).strip()
    if "mosek" in solver_name and not mosek_license_available():
        raise SolverNotAvailableError(
            f"Pyomo solver '{solver_name}' is not usable: the 'mosek' package "
            "is missing or no license could be checked out. MOSEK is "
            "optional; install it with 'pip install mosek' and place a "
            "license file in ~/mosek/mosek.lic (or set MOSEKLM_LICENSE_FILE)."
        )
    return get_pyomo_solver(name=solver_name, options=options)


def get_nlp_solver(
    name: str | None = None,
    options: dict[str, Any] | None = None,
    executable: str | os.PathLike[str] | None = None,
) -> Any:
    """Construct a configured Pyomo NLP solver.

    Parameters
    ----------
    name : str or None, optional
        Pyomo solver factory name. Defaults to
        :data:`pwlnet.config.DEFAULT_NLP_SOLVER` (``"ipopt"``). For
        ipopt the executable is located with :func:`find_ipopt`.
    options : dict or None, optional
        Solver options applied to the returned solver object.

    Returns
    -------
    object
        A Pyomo solver object ready for ``solve(model)``.

    Raises
    ------
    pwlnet.utils.exceptions.SolverNotAvailableError
        If pyomo is not installed or the requested NLP solver cannot be
        found. Ipopt is optional in PWLNet; callers that can fall back
        (or skip) should catch this exception.
    """
    return get_pyomo_solver(
        name=name or DEFAULT_NLP_SOLVER,
        options=options,
        executable=executable,
    )
