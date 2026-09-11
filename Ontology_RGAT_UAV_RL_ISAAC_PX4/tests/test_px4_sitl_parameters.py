import math
import pytest

from px4_sitl_parameters import configured_px4_parameters, px4_rc_script


def test_integer_parameters_remain_integers():
    parameter, = configured_px4_parameters({"COM_RC_IN_MODE": 4})

    assert parameter.name == "COM_RC_IN_MODE"
    assert parameter.value == 4


def test_rc_wrapper_sources_stock_script_before_setting_parameters(tmp_path):
    stock = tmp_path / "stock_rcS"
    parameters = configured_px4_parameters({
        "COM_RC_IN_MODE": 4,
        "COM_OF_LOSS_T": 5.0,
    })

    script = px4_rc_script(stock, parameters)

    assert script.index(f". {stock}") < script.index("param set COM_RC_IN_MODE 4")
    assert "param set COM_OF_LOSS_T 5" in script


@pytest.mark.parametrize("value", [True, "5", math.inf])
def test_invalid_parameter_values_are_rejected(value):
    with pytest.raises(ValueError):
        configured_px4_parameters({"COM_OF_LOSS_T": value})
