"""Package-wide defaults for PWLNet.

Small, dependency-light configuration constants used across the package.
Framework-specific configuration (Keras, Pyomo) lives in the respective
subpackages; only plain values and a logging helper belong here.
"""

from __future__ import annotations

import logging

import numpy as np

#: Default floating-point dtype for all numpy arrays in :mod:`pwlnet.core`.
DEFAULT_DTYPE: np.dtype = np.dtype(np.float64)

#: Default MILP/LP solver name (Pyomo solver factory string). HiGHS via the
#: APPSI interface is the guaranteed open-source default.
DEFAULT_MILP_SOLVER: str = "appsi_highs"

#: Default NLP solver name (Pyomo solver factory string). Ipopt is optional;
#: see :func:`pwlnet.solvers.solver_config.find_ipopt`.
DEFAULT_NLP_SOLVER: str = "ipopt"

#: Default conic (SOCP/MIQP/MISOCP) solver name (Pyomo solver factory
#: string). MOSEK is optional: it needs the ``mosek`` Python package and a
#: license file in the default location (``~/mosek/mosek.lic``) or pointed
#: at by ``MOSEKLM_LICENSE_FILE``.
DEFAULT_CONIC_SOLVER: str = "mosek_direct"

#: Environment variable that may point at an Ipopt executable.
IPOPT_ENV_VAR: str = "PWLNET_IPOPT"

#: Name of the root logger for the package.
LOGGER_NAME: str = "pwlnet"


def configure_logging(level: int | str = logging.INFO,
                      fmt: str = "%(asctime)s %(name)s %(levelname)s: %(message)s") -> logging.Logger:
    """Attach a basic stream handler to the ``pwlnet`` root logger.

    Library code never calls this itself (it only attaches a
    ``NullHandler``); applications and scripts may call it once to see
    PWLNet log output.

    Parameters
    ----------
    level : int or str, optional
        Logging level (e.g. ``logging.DEBUG`` or ``"DEBUG"``). Default
        ``logging.INFO``.
    fmt : str, optional
        Log record format string.

    Returns
    -------
    logging.Logger
        The configured ``pwlnet`` root logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger
