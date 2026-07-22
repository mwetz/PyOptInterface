"""PIPS-IPM++ backend for PyOptInterface.

PIPS-IPM++ is a block-structure exploiting interior-point solver: it uses a
doubly-bordered block-diagonal problem with diagonal blocks distributed across
MPI ranks. Recording the LP and its block annotation is handled via the
PIPS-IPM++ Python interface `pipsipmpppy`.

Assignment of blocks to variables is handled via the built-in
`VariableAttribute.Block`attribute (``0`` = root, ``1..N`` = leaf).
Equation blocks are derived automatically from the variables a constraint touches.

::
    import pyoptinterface as poi
    from pyoptinterface import pipsipmpp

    model = pipsipmpp.Model()
    cap = model.add_variable(lb=0.0, name="cap")
    model.set_variable_attribute(cap, poi.VariableAttribute.Block, 0)   # root
    gen = model.add_variable(lb=0.0, name="gen")
    model.set_variable_attribute(gen, poi.VariableAttribute.Block, 1)   # leaf 1
    ...
    model.set_objective(expr, poi.ObjectiveSense.Minimize)
    model.set_raw_parameter("LINEAR_ROOT_SOLVER", "mumps")
    model.set_raw_parameter("LINEAR_LEAF_SOLVER", "mumps")
    model.optimize()
    print(model.get_model_attribute(poi.ModelAttribute.ObjectiveValue))
    print(model.get_value(cap))

Run via ``mpirun -n <k> python your_script.py``. Only rank 0 builds the
model; every rank then calls `optimize` collectively. pipsipmpppy derives
the block structure on rank 0 and scatters the per-rank blocks
"""

from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np
import scipy.sparse as sp

from .attributes import (
    ConstraintAttribute,
    ModelAttribute,
    ResultStatusCode,
    TerminationStatusCode,
    VariableAttribute,
)
from .aml import make_variable_ndarray, make_variable_tupledict
from .comparison_constraint import ComparisonConstraint
from .core_ext import (
    ConstraintIndex,
    ConstraintSense,
    ConstraintType,
    ExprBuilder,
    ObjectiveSense,
    ScalarAffineFunction,
    VariableDomain,
    VariableIndex,
)
from .matrix import add_matrix_constraints
from .solver_common import (
    _direct_get_entity_attribute,
    _direct_get_model_attribute,
    _direct_set_entity_attribute,
    _direct_set_model_attribute,
)

INF = float("inf")


def _as_affine(expr) -> ScalarAffineFunction:
    if isinstance(expr, ScalarAffineFunction):
        return expr
    if isinstance(expr, (VariableIndex, ExprBuilder)):
        return ScalarAffineFunction(expr)
    if isinstance(expr, (int, float)):
        return ScalarAffineFunction(float(expr))
    raise ValueError(f"Unsupported linear expression of type {type(expr)}")


def _termination_map() -> dict:
    from pipsipmpppy import TerminationStatus as T

    C = TerminationStatusCode
    return {
        T.SUCCESSFUL_TERMINATION: C.OPTIMAL,
        T.NOT_FINISHED: C.OTHER_ERROR,
        T.MAX_ITS_EXCEEDED: C.ITERATION_LIMIT,
        T.TIMELIMIT: C.TIME_LIMIT,
        T.INFEASIBLE: C.INFEASIBLE,
        T.UNBOUNDED: C.DUAL_INFEASIBLE,
        T.READ_ERROR: C.INVALID_MODEL,
        T.DID_NOT_RUN: C.OPTIMIZE_NOT_CALLED,
        T.STOPPED_AFTER_PRESOLVE: C.INTERRUPTED,
        T.SLOW_CONVERGENCE: C.SLOW_PROGRESS,
        T.NUMERICAL: C.NUMERICAL_ERROR,
    }


def get_terminationstatus(model: "Model") -> TerminationStatusCode:
    if model._status is None:
        return TerminationStatusCode.OPTIMIZE_NOT_CALLED
    return _termination_map().get(model._status, TerminationStatusCode.OTHER_ERROR)


def _result_status(model: "Model") -> ResultStatusCode:
    if model._status is None:
        return ResultStatusCode.NO_SOLUTION
    if model._status.is_optimal:
        return ResultStatusCode.FEASIBLE_POINT
    if model._status.has_solution:
        return ResultStatusCode.UNKNOWN_RESULT_STATUS
    return ResultStatusCode.NO_SOLUTION


def get_dualstatus(model: "Model") -> ResultStatusCode:
    return _result_status(model)


def get_primalstatus(model: "Model") -> ResultStatusCode:
    return _result_status(model)


model_attribute_get_func_map = {
    ModelAttribute.ObjectiveSense: lambda m: m._sense,
    ModelAttribute.ObjectiveValue: lambda m: m._objective_value,
    ModelAttribute.TerminationStatus: get_terminationstatus,
    ModelAttribute.PrimalStatus: get_primalstatus,
    ModelAttribute.DualStatus: get_dualstatus,
    ModelAttribute.SolveTimeSec: lambda m: m._runtime,
    ModelAttribute.RawStatusString: lambda m: (
        "not optimized"
        if m._status is None
        else f"{m._status.name}: {m._status.description}"
    ),
    ModelAttribute.SolverName: lambda m: "PIPS-IPM++",
    ModelAttribute.Silent: lambda m: m._silent,
}

model_attribute_set_func_map = {
    ModelAttribute.ObjectiveSense: lambda m, v: setattr(m, "_sense", v),
    ModelAttribute.Silent: lambda m, v: setattr(m, "_silent", v),
}

variable_attribute_get_func_map = {
    VariableAttribute.Value: lambda m, v: m._value[v.index],
    VariableAttribute.LowerBound: lambda m, v: m._lb[v.index],
    VariableAttribute.UpperBound: lambda m, v: m._ub[v.index],
    VariableAttribute.Name: lambda m, v: m._vname[v.index],
    VariableAttribute.Block: lambda m, v: m._block[v.index],
}

variable_attribute_set_func_map = {
    VariableAttribute.LowerBound: lambda m, v, val: m._lb.__setitem__(v.index, val),
    VariableAttribute.UpperBound: lambda m, v, val: m._ub.__setitem__(v.index, val),
    VariableAttribute.Name: lambda m, v, val: m._vname.__setitem__(v.index, val),
    VariableAttribute.Block: lambda m, v, val: m._block.__setitem__(v.index, int(val)),
}

constraint_attribute_get_func_map = {
    ConstraintAttribute.Name: lambda m, c: m._cname[c.index],
    ConstraintAttribute.Primal: lambda m, c: m._constraint_primal(c.index),
    ConstraintAttribute.Dual: lambda m, c: m._dual[c.index],
    ConstraintAttribute.Block: lambda m, c: m._constraint_block(c.index),
}

constraint_attribute_set_func_map = {
    ConstraintAttribute.Name: lambda m, c, val: m._cname.__setitem__(c.index, val),
    ConstraintAttribute.Block: lambda m, c, val: m._cblock.__setitem__(
        c.index, int(val)
    ),
}


class Model:
    def __init__(self, comm=None, jit=None):
        if comm is None:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        self._comm = comm

        self._lb: list[float] = []
        self._ub: list[float] = []
        self._vname: list[str] = []
        self._block: list[int] = []
        self._value: list[float] = []
        self._obj: dict[int, float] = {}
        self._obj_const: float = 0.0
        self._sense: ObjectiveSense = ObjectiveSense.Minimize
        self._crows: list[ScalarAffineFunction] = []
        self._csense: list[ConstraintSense] = []
        self._crhs: list[float] = []
        self._cname: list[str] = []
        self._cblock: dict[int, int] = {}
        self._options: dict[str, object] = {}
        self._silent: bool = False
        self._status = None
        self._objective_value: Optional[float] = None
        self._runtime: float = 0.0
        self._dual: list[float] = []
        self._crow_slot: list[tuple[bool, int]] = []

    def add_variable(
        self,
        domain: VariableDomain = VariableDomain.Continuous,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        name: str = "",
        start: Optional[float] = None,
        block: Optional[int] = None,
    ) -> VariableIndex:
        if domain != VariableDomain.Continuous:
            raise ValueError("PIPS-IPM++ only supports linear models at the moment")
        index = len(self._lb)
        self._lb.append(-INF if lb is None else float(lb))
        self._ub.append(INF if ub is None else float(ub))
        self._vname.append(name)
        self._block.append(0 if block is None else int(block))
        self._value.append(0.0)
        return VariableIndex(index)

    add_variables = make_variable_tupledict
    add_m_variables = make_variable_ndarray
    add_m_linear_constraints = add_matrix_constraints

    @overload
    def add_linear_constraint(
        self,
        expr: Union[VariableIndex, ScalarAffineFunction, ExprBuilder],
        sense: ConstraintSense,
        rhs: float,
        name: str = "",
    ) -> ConstraintIndex: ...

    @overload
    def add_linear_constraint(
        self, con: ComparisonConstraint, name: str = ""
    ) -> ConstraintIndex: ...

    def add_linear_constraint(self, arg, *args, **kwargs) -> ConstraintIndex:
        if isinstance(arg, ComparisonConstraint):
            return self._add_linear_constraint(
                arg.lhs, arg.sense, arg.rhs, *args, **kwargs
            )
        return self._add_linear_constraint(arg, *args, **kwargs)

    def _add_linear_constraint(
        self, expr, sense: ConstraintSense, rhs: float, name: str = ""
    ) -> ConstraintIndex:
        saf = _as_affine(expr)
        index = len(self._crows)
        self._crows.append(saf)
        self._csense.append(sense)
        self._crhs.append(float(rhs))
        self._cname.append(name)
        return ConstraintIndex(ConstraintType.Linear, index)

    def set_objective(
        self, expr, sense: ObjectiveSense = ObjectiveSense.Minimize
    ) -> None:
        saf = _as_affine(expr)
        self._obj = {}
        for var, coef in zip(saf.variables, saf.coefficients):
            self._obj[int(var)] = self._obj.get(int(var), 0.0) + float(coef)
        self._obj_const = float(saf.constant) if saf.constant is not None else 0.0
        self._sense = sense

    @staticmethod
    def supports_variable_attribute(attribute, settable=False):
        m = (
            variable_attribute_set_func_map
            if settable
            else variable_attribute_get_func_map
        )
        return attribute in m

    @staticmethod
    def supports_model_attribute(attribute, settable=False):
        m = model_attribute_set_func_map if settable else model_attribute_get_func_map
        return attribute in m

    @staticmethod
    def supports_constraint_attribute(attribute, settable=False):
        m = (
            constraint_attribute_set_func_map
            if settable
            else constraint_attribute_get_func_map
        )
        return attribute in m

    def get_variable_attribute(self, variable, attribute: VariableAttribute):
        def e(a):
            raise ValueError(f"Unknown variable attribute to get: {a}")

        return _direct_get_entity_attribute(
            self, variable, attribute, variable_attribute_get_func_map, e
        )

    def set_variable_attribute(self, variable, attribute: VariableAttribute, value):
        def e(a):
            raise ValueError(f"Unknown variable attribute to set: {a}")

        _direct_set_entity_attribute(
            self, variable, attribute, value, variable_attribute_set_func_map, e
        )

    def get_model_attribute(self, attribute: ModelAttribute):
        def e(a):
            raise ValueError(f"Unknown model attribute to get: {a}")

        return _direct_get_model_attribute(
            self, attribute, model_attribute_get_func_map, e
        )

    def set_model_attribute(self, attribute: ModelAttribute, value):
        def e(a):
            raise ValueError(f"Unknown model attribute to set: {a}")

        _direct_set_model_attribute(
            self, attribute, value, model_attribute_set_func_map, e
        )

    def get_constraint_attribute(self, constraint, attribute: ConstraintAttribute):
        def e(a):
            raise ValueError(f"Unknown constraint attribute to get: {a}")

        return _direct_get_entity_attribute(
            self, constraint, attribute, constraint_attribute_get_func_map, e
        )

    def set_constraint_attribute(
        self, constraint, attribute: ConstraintAttribute, value
    ):
        def e(a):
            raise ValueError(f"Unknown constraint attribute to set: {a}")

        _direct_set_entity_attribute(
            self, constraint, attribute, value, constraint_attribute_set_func_map, e
        )

    def get_value(self, variable: VariableIndex) -> float:
        return self._value[variable.index]

    def number_of_variables(self) -> int:
        return len(self._lb)

    def number_of_constraints(
        self, type: ConstraintType = ConstraintType.Linear
    ) -> int:
        if type != ConstraintType.Linear:
            return 0
        return len(self._crows)

    def set_raw_parameter(self, name: str, value) -> None:
        self._options[name] = value

    def get_raw_parameter(self, name: str):
        return self._options[name]

    def optimize(self, options: Optional[dict] = None) -> None:
        from pipsipmpppy import solve

        opts = dict(self._options)
        if options:
            opts.update(options)

        # Model is built on rank 0 only, other ranks only solve
        problem = self._build_problem() if self._comm.Get_rank() == 0 else None
        result = solve(problem, self._comm, options=opts)

        self._status = result.status
        obj = result.objective
        if self._sense == ObjectiveSense.Maximize:
            obj = -obj
        self._objective_value = obj

        self._runtime = float(result.runtime)
        if result.primal is not None:
            self._value = [float(x) for x in result.primal]
        if result.dual_eq is not None:
            sign = -1.0 if self._sense == ObjectiveSense.Maximize else 1.0
            self._dual = [
                sign * float((result.dual_eq if is_eq else result.dual_ineq)[row])
                for is_eq, row in self._crow_slot
            ]
        self._comm.Barrier()

    def _obj_vector(self) -> np.ndarray:
        nv = len(self._lb)
        c = np.zeros(nv, dtype=float)
        for j, coef in self._obj.items():
            c[j] = coef
        if self._sense == ObjectiveSense.Maximize:
            c = -c
        return c

    def _split_constraints(self):
        eq_rows, eq_rhs = [], []
        iq_rows, iq_low, iq_upp = [], [], []
        self._crow_slot = []
        for saf, sense, rhs in zip(self._crows, self._csense, self._crhs):
            const = float(saf.constant) if saf.constant is not None else 0.0
            terms = list(
                zip((int(v) for v in saf.variables), map(float, saf.coefficients))
            )
            b = rhs - const
            if sense == ConstraintSense.Equal:
                self._crow_slot.append((True, len(eq_rows)))
                eq_rows.append(terms)
                eq_rhs.append(b)
            elif sense == ConstraintSense.LessEqual:
                self._crow_slot.append((False, len(iq_rows)))
                iq_rows.append(terms)
                iq_low.append(-INF)
                iq_upp.append(b)
            elif sense == ConstraintSense.GreaterEqual:
                self._crow_slot.append((False, len(iq_rows)))
                iq_rows.append(terms)
                iq_low.append(b)
                iq_upp.append(INF)
            else:
                raise ValueError(f"Unsupported constraint sense: {sense}")
        return eq_rows, eq_rhs, iq_rows, iq_low, iq_upp

    def _build_problem(self):
        from pipsipmpppy import StructuredProblem

        nv = len(self._lb)
        n_blocks = max(self._block) if self._block else 0

        def to_csr(rows):
            data, ind, ptr = [], [], [0]
            for terms in rows:
                for j, coef in terms:
                    ind.append(j)
                    data.append(coef)
                ptr.append(len(ind))
            return sp.csr_array(
                (
                    np.asarray(data, float),
                    np.asarray(ind, int),
                    np.asarray(ptr, int),
                ),
                shape=(len(rows), nv),
            )

        objconst = (
            -self._obj_const
            if self._sense == ObjectiveSense.Maximize
            else self._obj_const
        )

        eq_rows, eq_rhs, iq_rows, iq_low, iq_upp = self._split_constraints()
        return StructuredProblem(
            n_blocks=n_blocks,
            var_block=np.asarray(self._block, dtype=np.int64),
            c=self._obj_vector(),
            xlow=np.asarray(self._lb, float),
            xupp=np.asarray(self._ub, float),
            A_eq=to_csr(eq_rows),
            b_eq=np.asarray(eq_rhs, float),
            A_ineq=to_csr(iq_rows),
            ineq_low=np.asarray(iq_low, float),
            ineq_upp=np.asarray(iq_upp, float),
            objconst=objconst,
        )

    def _constraint_primal(self, index: int) -> float:
        saf = self._crows[index]
        return float(
            sum(
                float(coef) * self._value[int(var)]
                for var, coef in zip(saf.variables, saf.coefficients)
            )
        )

    def _constraint_block(self, index: int) -> int:
        if index in self._cblock:
            return self._cblock[index]
        saf = self._crows[index]
        leaves = {
            self._block[int(v)] for v in saf.variables if self._block[int(v)] != 0
        }
        if not leaves:
            return 0
        if len(leaves) == 1:
            return next(iter(leaves))
        return -1
