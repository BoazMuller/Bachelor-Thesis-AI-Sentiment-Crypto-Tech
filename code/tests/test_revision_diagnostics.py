import numpy as np
import pandas as pd
import pytest
from thesis.modeling.diagnostics import one_lag_granger_test


def test_granger_direction_and_effective_sample():
    rng = np.random.default_rng(312)
    x = rng.normal(size=400)
    y = np.r_[0, 2*x[:-1]] + rng.normal(scale=.3, size=400)
    frame = pd.DataFrame({'x': x, 'y': y})
    forward = one_lag_granger_test(frame, 'y', 'x')
    reverse = one_lag_granger_test(frame, 'x', 'y')
    assert forward['p_value'] < .001
    assert reverse['p_value'] > .05
    assert forward['nobs'] == 399
    assert forward['df_denom'] == 396


def test_granger_rejects_internal_missing_date_but_accepts_initial_missingness():
    rng = np.random.default_rng(122)
    frame = pd.DataFrame(rng.normal(size=(100, 2)), columns=['x', 'y'])
    frame.loc[:3, 'x'] = np.nan
    assert one_lag_granger_test(frame, 'y', 'x')['nobs'] == 95
    frame.loc[50, 'x'] = np.nan
    with pytest.raises(ValueError, match='contiguous'):
        one_lag_granger_test(frame, 'y', 'x')
