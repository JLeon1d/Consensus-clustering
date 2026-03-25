"""Utilities for optional Ray parallelization."""

import warnings
from typing import Optional

_ray_initialized = False


def is_ray_available() -> bool:
    """
    Check if Ray is installed and can be imported.

    Returns
    -------
    bool
        True if Ray is available, False otherwise
    """
    try:
        import ray
        return True
    except ImportError:
        return False


def init_ray_if_needed(use_ray: bool = True, **ray_init_kwargs) -> bool:
    """
    Initialize Ray if requested and available.
    """
    global _ray_initialized

    if not use_ray:
        return False

    if not is_ray_available():
        warnings.warn(
            "Ray is not installed. Install with: pip install ray>=2.9.0. "
            "Falling back to sequential execution.",
            UserWarning
        )
        return False

    import ray

    if not ray.is_initialized():
        try:
            ray.init(**ray_init_kwargs)
            _ray_initialized = True
        except Exception as e:
            warnings.warn(
                f"Failed to initialize Ray: {e}. Falling back to sequential execution.",
                UserWarning
            )
            return False

    return True


def shutdown_ray_if_initialized():
    """Shutdown Ray if it was initialized by this module."""
    global _ray_initialized

    if _ray_initialized:
        try:
            import ray
            ray.shutdown()
            _ray_initialized = False
        except Exception:
            pass


def get_ray_status() -> dict:
    """
    Get current Ray status information.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'available': bool, whether Ray is installed
        - 'initialized': bool, whether Ray is currently initialized
        - 'num_cpus': int or None, number of CPUs if initialized
    """
    status = {
        'available': is_ray_available(),
        'initialized': False,
        'num_cpus': None
    }

    if status['available']:
        import ray
        status['initialized'] = ray.is_initialized()
        if status['initialized']:
            try:
                status['num_cpus'] = int(ray.available_resources().get('CPU', 0))
            except Exception:
                pass

    return status