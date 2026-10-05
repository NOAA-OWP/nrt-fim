import abc

import torch
from numpy.typing import NDArray
from omegaconf import DictConfig


class BMIBase(abc.ABC):
    """Simplified BMI interface for inference-only runs."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = None

    @abc.abstractmethod
    def initialize(self):
        """Build the model, load weights, move to device."""
        ...

    @abc.abstractmethod
    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform one forward step. Returns model output tensor."""
        ...

    @abc.abstractmethod
    def finalize(self):
        """Any cleanup/summary (optional)."""
        pass

    #############
    # Bmi functions which have a reasonable "default" implementation
    #############
    def get_component_name(self) -> str:
        """Name of this BMI module component.

        Returns
        -------
            str: Model Name
        """
        return self.__class__.__name__

    # Some "optional" functions which have a reasonable default implementation
    # as long as get_value_ptr is implemented and the typing is adhered to
    # TODO: should these try/except to catch the get_value_ptr unimplmented error
    # and raise a more tailored exception indicating the problem/solution?

    def get_value(self, name: str, dest: NDArray) -> NDArray:
        """Gets value ptrt"""
        dest[:] = self.get_value_ptr(name)
        return dest

    def get_var_nbytes(self, name: str) -> int:
        """Get the number of total bytes required to represent the variable.

        Args:
            name (str): Name of variable.

        Returns
        -------
            int: Size of data array in bytes.
        """
        return self.get_value_ptr(name).nbytes

    def get_var_type(self, name: str) -> str:
        """Data type of the variable.

           If the variable is an array, this is the type of a single
           element of the array.

        Args:
            name (str): Name of variable.

        Returns
        -------
            str: Data type.
        """
        return str(self.get_value_ptr(name).dtype)

    ###############
    # BMI functions which may be considered "optional" for a minimally functioning
    # BMI implementation
    ###############
    def update_until(self, time: float) -> None:
        """Update model from current_time until current_time + time

        Args:
            time (float): duration of time to advance model till
        """
        raise NotImplementedError()

    # BMI Variable Information Functions
    def get_input_item_count(self) -> int:
        """Number of model input variables

        Returns
        -------
            int: number of input variables
        """
        raise NotImplementedError()

    def get_input_var_names(self) -> tuple[str]:
        """The names of each input variables

        Returns
        -------
            Tuple[str]: iterable tuple of input variable names
        """
        raise NotImplementedError()

    def get_output_item_count(self) -> int:
        """Number of model output variables

        Returns
        -------
            int: number of output variables
        """
        raise NotImplementedError()

    def get_output_var_names(self) -> tuple[str]:
        """The names of each output variable

        Returns
        -------
            Tuple[str]: iterable tuple of output variable names
        """
        raise NotImplementedError()

    # BMI Variable Information Functions
    def get_var_grid(self, name: str) -> int:
        """Get the grid identiferier associated with a given variable

        Args:
            name (str): name of the variable

        Raises
        ------
            UnknownBMIVariable: name is not recognized, grid unknown

        Returns
        -------
            int: grid identifier associated with @p name
        """
        raise NotImplementedError()

    def get_var_itemsize(self, name: str) -> int:
        """Size, in bytes, of a single element of the variable name

        Args:
            name (str): variable name

        Returns
        -------
            int: number of bytes representing a single variable of @p name
        """
        raise NotImplementedError()

    def get_var_location(self, name: str) -> str:
        """Location of the variable relative to the grid

        Args:
            name (str): name of the BMI variable

        Raises
        ------
            UnknownBMIVariable: name is not recognized, location unknown

        Returns
        -------
            str: location on the grid, e.g. node, face
        """
        raise NotImplementedError()

    def get_var_units(self, name: str) -> str:
        """Get units of the given variable

        Args:
            name (str): variable name

        Raises
        ------
            UnknownBMIVariable: name is not recognized, units unknown

        Returns
        -------
            str: units
        """
        raise NotImplementedError()

    def get_current_time(self) -> float:
        """Gets the current time value"""
        raise NotImplementedError()

    def get_end_time(self) -> float:
        """Gets the end time"""
        raise NotImplementedError()

    # BMI grid functions
    def get_grid_edge_count(self, grid: int) -> int:
        """Gets number of grid edges"""
        raise NotImplementedError()

    def get_grid_edge_nodes(self, grid: int, edge_nodes: NDArray) -> NDArray:
        """Gets the nodes in the edges"""
        raise NotImplementedError()

    def get_grid_face_count(self, grid: int) -> int:
        """Gets number of grids?"""
        raise NotImplementedError()

    def get_grid_face_edges(self, grid: int, face_edges: NDArray) -> NDArray:
        """Gets grid_face_edge"""
        raise NotImplementedError()

    def get_grid_face_nodes(self, grid: int, face_nodes: NDArray) -> NDArray:
        """Gets grid face nodes"""
        raise NotImplementedError()

    def get_grid_node_count(self, grid: int) -> int:
        """Gets number of grid nodes"""
        raise NotImplementedError()

    def get_grid_nodes_per_face(self, grid: int, nodes_per_face: NDArray) -> NDArray:
        """Gets grid_nodes_per_face"""
        raise NotImplementedError()

    def get_grid_origin(self, grid: int, origin: NDArray) -> NDArray:
        """Gets grid origin"""
        raise NotImplementedError()

    def get_grid_rank(self, grid: int) -> int:
        """Gets grid rank"""
        raise NotImplementedError()

    def get_grid_shape(self, grid: int, shape: NDArray) -> NDArray:
        """Gets grid shape"""
        raise NotImplementedError()

    def get_grid_size(self, grid: int) -> int:
        """Gets size of a grid"""
        raise NotImplementedError()

    def get_grid_spacing(self, grid: int, spacing: NDArray) -> NDArray:
        """Gets grid_spacing"""
        raise NotImplementedError()

    def get_grid_type(self, grid: int) -> str:
        """Gets grid type"""
        raise NotImplementedError()

    def get_grid_x(self, grid: int, x: NDArray) -> NDArray:
        """Gets x value of a grid"""
        raise NotImplementedError()

    def get_grid_y(self, grid: int, y: NDArray) -> NDArray:
        """Gets y vlue of a grid"""
        raise NotImplementedError()

    def get_grid_z(self, grid: int, z: NDArray) -> NDArray:
        """Gets z value of a grid"""
        raise NotImplementedError()

    def get_start_time(self) -> float:
        """Gets start time"""
        raise NotImplementedError()

    def get_time_step(self) -> float:
        """Gets the time step"""
        raise NotImplementedError()

    def get_time_units(self) -> str:
        """Gets time unit"""
        raise NotImplementedError()

    # BMI get/set
    def get_value_ptr(self, name: str) -> NDArray:
        """Gets value_ptr"""
        raise NotImplementedError

    def get_value_at_indices(self, name: str, dest: NDArray, inds: NDArray) -> NDArray:
        """Gets value at indices"""
        raise NotImplementedError()

    def set_value(self, name: str, src: NDArray) -> None:
        """Sets value"""
        raise NotImplementedError()

    def set_value_at_indices(self, name: str, inds: NDArray, src: NDArray) -> None:
        """Sets value at indices"""
        raise NotImplementedError()
