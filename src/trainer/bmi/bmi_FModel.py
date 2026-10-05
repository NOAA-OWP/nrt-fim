import logging
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig

from trainer import FModel
from trainer.bmi.bmi_base import BMIBase

logger = logging.getLogger(__name__)

__all__ = ["FModelBMI"]


class FModelBMI(BMIBase):
    """running FModel with BMI"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        ## build and move to device
        # self.model = self.initialize()

        self.input_names = (
            "precip",
            "air_temp",
            "solar_shortwave",
            "soil_sm_layer1",
            "soil_sm_layer2",
            "soil_sm_layer3",
            "soil_sm_layer4",
            "SNOWH",
            "ugd_runoff",
        )
        self.output_names = "FIM_sim_perc"

        self.input: dict[Any, Any] = dict(
            zip(self.input_names, [0 for i in range(len(self.input_names))], strict=False)
        )
        self.output: dict[Any, Any] = dict(
            zip(self.output_names, [0 for i in range(len(self.output_names))], strict=False)
        )

    def initialize(self):
        """Building the model. build and send to device"""
        self.model = FModel(
            num_classes=int(self.cfg.model.num_classes),
            in_channels=int(self.cfg.model.in_channels),
            device=self.device,
        )
        # load checkpoint if provided --> should be umcommented when the first model has been  trained
        # ckpt = self.cfg.params.get("checkpoint")
        # if ckpt and os.path.exists(ckpt):
        #     state = torch.load(ckpt, map_location=self.device)
        #     self.model.load_state_dict(state["model_state"])
        self.model.eval()
        return self.model

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward-runs the model on a single event.

        Note: self.model.eval() has ben don in initialize()
        inputs: (batch, channels, H, W)
        returns: (batch, 1, H, W)
        """
        with torch.no_grad():
            return self.model(inputs.to(self.device))

    def finalize(self):
        """Nothing special for inference"""
        pass

    def get_component_name(self) -> str:
        """Name of this BMI module component.

        Returns
        -------
            str: Model Name
        """
        return "F1-trainer"

    def get_input_item_count(self) -> int:
        """Number of model input variables

        Returns
        -------
            int: number of input variables
        """
        return len(self.input_names)

    def get_input_var_names(self) -> tuple[str, ...]:  # type: ignore[override]
        """The names of each input variables

        Returns
        -------
            tuple[str, ...]: iterable tuple of input variable names
        """
        return self.input_names

    def get_output_item_count(self) -> int:
        """Number of model output variables

        Returns
        -------
            int: number of output variables
        """
        return len(self.output_names)

    def get_output_var_names(self) -> tuple[str, ...]:  # type: ignore[override]
        """The names of each output variable

        Returns
        -------
            tuple[str, ...]: iterable tuple of output variable names
        """
        return self.output_names

    # BMI Variable Query
    def set_value(self, name: str, src: Any) -> None:
        """Sets and input or output value

        Args:
            name (str): name of value
            src (Any): value to set

        Raises
        ------
            ValueError: If name does not exist
        """
        if name in self.output_names:
            self.output[name] = src
        elif name in self.input_names:
            self.input[name] = src
        else:
            raise ValueError(
                f"Variable {name} does not exist input or output variables.  User getters to view options."
            )

    def get_value(self, name: str, dest: NDArray) -> NDArray:
        """_Copies_ a variable's np.np.ndarray into `dest` and returns `dest`."""
        value = self.get_value_ptr(name)
        try:
            if not isinstance(value, np.ndarray):
                dest[:] = np.array(value).flatten()
            else:
                dest[:] = self.get_value_ptr(name).flatten()
        except Exception as e:
            raise RuntimeError(f"Could not return value {name} as flattened array") from e

        return dest

    def get_value_ptr(self, name: str) -> NDArray:
        """Gets value in native form if exists in inputs or outputs"""
        try:
            return self.output[name]
        except KeyError:
            return self.input[name]
        except KeyError as e:  # NOQA: B025
            raise KeyError(f"{name} is not a known variable") from e

    def get_var_itemsize(self, name: str) -> int:
        """Size, in bytes, of a single element of the variable name

        Args:
            name (str): variable name

        Returns
        -------
            int: number of bytes representing a single variable of @p name
        """
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:
        """Size, in nbytes, of a single element of the variable name

        Args:
            name (str): Name of variable.

        Returns
        -------
            int: Size of data array in bytes.
        """
        return self.get_value_ptr(name).nbytes

    def get_var_type(self, name: str) -> str:
        """Data type of variable.

        Args:
            name (str): Name of variable.

        Returns
        -------
            str: Data type.
        """
        return str(self.get_value_ptr(name).dtype)
