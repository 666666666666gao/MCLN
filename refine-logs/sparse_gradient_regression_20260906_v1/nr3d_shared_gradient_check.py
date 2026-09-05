"""Numerical parity check for the measured native backward variability."""


def shared_gradient_comparison(reference, current):
    import torch

    difference = current.double() - reference.double()
    relative_l2 = float(difference.norm() / reference.double().norm())
    return {
        'relative_l2': relative_l2,
        'relative_l2_limit': 1e-4,
        'passed': relative_l2 <= 1e-4,
        'max_abs_difference': float(difference.abs().max()),
        'elementwise_atol1e6_rtol1e5': torch.allclose(reference, current, atol=1e-6, rtol=1e-5),
    }
