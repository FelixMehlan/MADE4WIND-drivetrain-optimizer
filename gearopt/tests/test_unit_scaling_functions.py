
import numpy as np
from gearopt.util.scale_variables import scale_variables
from gearopt.util.unscale_variables import unscale_variables

def test_scale_variables_basic():
    x = np.array([5, 15, 50])
    lb = np.array([0, 10, 40])
    ub = np.array([10, 20, 60])

    x_scaled, lb_s, ub_s, scale = scale_variables(x, lb, ub)

    # Expected results
    expected_scaled = np.array([0.5, 0.5, 0.5])
    expected_scale = np.array([10, 10, 20])

    assert np.allclose(x_scaled, expected_scaled), f"x_scaled incorrect: {x_scaled}"
    assert np.allclose(scale, expected_scale), f"scale incorrect: {scale}"
    assert np.allclose(lb_s, np.zeros(3))
    assert np.allclose(ub_s, np.ones(3))

def test_scale_and_unscale_inverse():
    # Random test values
    x = np.array([2.5, 12.0, 80.0])
    lb = np.array([0.0, 10.0, 50.0])
    ub = np.array([10.0, 20.0, 100.0])

    x_scaled, _, _, _ = scale_variables(x, lb, ub)
    x_unscaled = unscale_variables(x_scaled, lb, ub)

    assert np.allclose(x, x_unscaled), f"scale/unscale mismatch: {x_unscaled} vs {x}"

def test_fixed_variable_case():
    # When ub == lb
    x = np.array([10])
    lb = np.array([10])
    ub = np.array([10])

    x_scaled, lb_s, ub_s, scale = scale_variables(x, lb, ub)

    assert scale[0] == 1.0, "Scale should be forced to 1 when ub == lb"
    assert x_scaled[0] == 0.0, "Scaled value should be 0 when ub == lb"
    assert ub_s[0] == 0.0, "Upper scaled bound should be 0 when variable is fixed"
