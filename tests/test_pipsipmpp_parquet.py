"""Writing an annotated model out as parquet for a later or remote solve."""

import pytest

import pyoptinterface as poi

pipsipmpp = pytest.importorskip("pipsipmpp")
pytest.importorskip("pyarrow")

# PyOptInterface's backend module carries the same name as the solver interface
# package it drives, so the backend is bound under a name of its own here
from pyoptinterface import pipsipmpp as backend  # noqa: E402


def _two_block_model(sense=poi.ObjectiveSense.Minimize):
    """Root variable ``y`` coupling two single-variable leaf blocks."""
    model = backend.Model()
    y = model.add_variable(lb=0.0, ub=10.0, name="y", block=0)
    x1 = model.add_variable(lb=0.0, name="x1", block=1)
    x2 = model.add_variable(lb=0.0, name="x2", block=2)

    model.add_linear_constraint(x1, poi.Geq, 2.0, name="demand1")
    model.add_linear_constraint(x2, poi.Geq, 3.0, name="demand2")
    model.add_linear_constraint(x1 - y, poi.Leq, 0.0, name="cap1")
    model.add_linear_constraint(x2 - y, poi.Leq, 0.0, name="cap2")
    model.set_objective(y + 0.1 * x1 + 0.1 * x2, sense)
    return model


@pytest.mark.parametrize("layout", ["monolithic", "distributed"])
def test_write_parquet_exports_both_layouts(tmp_path, layout):
    stem = _two_block_model().write_parquet(tmp_path / layout, layout=layout)

    manifest = pipsipmpp.read_manifest(stem)
    assert manifest["layout"] == layout
    assert manifest["n_blocks"] == 3  # the root counts alongside the two leaves
    assert [block["n"] for block in manifest["blocks"]] == [1, 1, 1]


def test_both_layouts_describe_the_same_problem(tmp_path):
    whole = pipsipmpp.read_manifest(
        _two_block_model().write_parquet(tmp_path / "whole")
    )
    split = pipsipmpp.read_manifest(
        _two_block_model().write_parquet(tmp_path / "split", layout="distributed")
    )

    assert (whole["n_rows"], whole["n_cols"]) == (split["n_rows"], split["n_cols"])
    keys = ("n", "my", "mz", "myl", "mzl")
    assert [[b[k] for k in keys] for b in whole["blocks"]] == [
        [b[k] for k in keys] for b in split["blocks"]
    ]


def test_rows_and_columns_carry_their_model_names(tmp_path):
    stem = _two_block_model().write_parquet(tmp_path / "named", names=True)
    names = pipsipmpp.read_names(stem)

    assert sorted(names["cols"]) == ["x1", "x2", "y"]
    assert sorted(names["rows"]) == ["cap1", "cap2", "demand1", "demand2"]


def test_names_can_be_left_out(tmp_path):
    stem = _two_block_model().write_parquet(tmp_path / "plain", names=False)
    assert pipsipmpp.read_names(stem) == {}

    # leaving the names out must not change the structure
    named = _two_block_model().write_parquet(tmp_path / "named", names=True)
    keys = ("n", "my", "mz", "myl", "mzl")
    assert [
        [b[k] for k in keys] for b in pipsipmpp.read_manifest(stem)["blocks"]
    ] == [[b[k] for k in keys] for b in pipsipmpp.read_manifest(named)["blocks"]]


def test_names_line_up_with_the_rows_they_label(tmp_path):
    """A name on the wrong row would mislabel a plot without failing anywhere."""
    pq = pytest.importorskip("pyarrow.parquet")

    stem = _two_block_model().write_parquet(tmp_path / "aligned", names=True)
    cols = pq.read_table(f"{stem}.cols.parquet")
    rows = pq.read_table(f"{stem}.rows.parquet")
    amat = pq.read_table(f"{stem}.amat.parquet")

    col_names = cols["name"].to_pylist()
    row_names = rows["name"].to_pylist()
    entries = list(zip(amat["row"].to_pylist(), amat["col"].to_pylist()))

    # cap1 is `x1 - y <= 0`, so it touches exactly y and x1
    row = row_names.index("cap1")
    assert {col_names[c] for r, c in entries if r == row} == {"y", "x1"}

    # y couples the two leaves, so it is the root variable
    partition = cols["partition"].to_pylist()
    assert partition[col_names.index("y")] == 1
    assert partition[col_names.index("x1")] != 1


def test_a_maximisation_records_its_sense(tmp_path):
    model = _two_block_model(poi.ObjectiveSense.Maximize)
    stem = model.write_parquet(tmp_path / "maxi")
    # the problem holds minimisation costs; objcoef says how the model stated them
    assert pipsipmpp.read_manifest(stem)["objcoef"] == -1.0


def test_unknown_layout_is_reported(tmp_path):
    with pytest.raises(ValueError, match="unknown layout"):
        _two_block_model().write_parquet(tmp_path / "nope", layout="blockwise")


@pytest.fixture
def options_file(tmp_path):
    """A settings file, as a repeatable run would keep on disk."""
    path = tmp_path / "base.opt"
    path.write_text(
        "# base configuration\n"
        "SCALER                       geometricmean\n"
        "PRESOLVE_BOUND_STR_MAX_ITER  7\n"
    )
    return path


def _solved(model):
    return model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)


def test_options_file_from_the_constructor(options_file):
    model = _two_block_model()
    model.set_options_file(options_file)
    assert model.get_options_file() == options_file
    model.optimize()
    # x1 >= 2, x2 >= 3, y >= max(x) = 3  ->  obj = 3 + 0.1*5 = 3.5
    assert _solved(model) == pytest.approx(3.5, abs=1e-5)


def test_options_override_the_file(options_file):
    """The file is the base; options set on the model win."""
    model = _two_block_model()
    model.set_options_file(options_file)
    model.set_raw_parameter("SCALER", "none")
    model.set_raw_parameter("PRESOLVE_BOUND_STR_MAX_ITER", 2)
    model.optimize()
    assert _solved(model) == pytest.approx(3.5, abs=1e-5)


def test_optimize_can_name_a_file_of_its_own(options_file):
    model = _two_block_model()
    model.optimize(options_file=options_file)
    assert _solved(model) == pytest.approx(3.5, abs=1e-5)


def test_a_missing_options_file_is_reported(tmp_path):
    model = _two_block_model()
    with pytest.raises(FileNotFoundError, match="options file"):
        model.optimize(options_file=tmp_path / "absent.opt")


def test_names_are_off_by_default(tmp_path):
    """Producing names costs a pass over the model, so they are opt-in."""
    stem = _two_block_model().write_parquet(tmp_path / "default")
    assert pipsipmpp.read_names(stem) == {}


def _solution_beside(stem, sense=1.0):
    """Write a solution whose values are the global index they belong to."""
    from pipsipmpp import TerminationStatus
    from pipsipmpp.flat import solver_order, write_solution

    order = solver_order(stem)
    write_solution(
        stem,
        order=order,
        status=TerminationStatus.SUCCESSFUL_TERMINATION,
        objective=sense * 12.5,
        runtime=0.0,
        iterations=7,
        primal=[float(c) for c in order.cols],
        reduced_costs=[float(c) for c in order.cols],
        dual_eq=[float(r) for r in order.eq_rows],
        dual_ineq=[float(r) for r in order.ineq_rows],
    )


def _mixed_model(sense=poi.ObjectiveSense.Minimize):
    """Two leaf blocks, one equality and two inequalities, so both duals matter."""
    model = backend.Model()
    y = model.add_variable(lb=0.0, name="y", block=0)
    x1 = model.add_variable(lb=0.0, name="x1", block=1)
    x2 = model.add_variable(lb=0.0, name="x2", block=2)

    eq = model.add_linear_constraint(x1 + x2, poi.Eq, 5.0, name="total")
    c1 = model.add_linear_constraint(x1 - y, poi.Leq, 0.0, name="cap1")
    c2 = model.add_linear_constraint(x2 - y, poi.Leq, 0.0, name="cap2")
    model.set_objective(y + 0.1 * x1 + 0.1 * x2, sense)
    return model, (y, x1, x2), (eq, c1, c2)


def test_read_parquet_solution_lands_on_the_model(tmp_path):
    model, (y, x1, x2), (eq, c1, c2) = _mixed_model()
    stem = model.write_parquet(tmp_path / "model", layout="distributed")
    _solution_beside(stem)

    model.read_parquet_solution(stem)

    # variable j carries the value j, so nothing was reordered on the way back
    assert [model.get_value(v) for v in (y, x1, x2)] == [0.0, 1.0, 2.0]
    assert model.get_model_attribute(poi.ModelAttribute.ObjectiveValue) == 12.5
    assert (
        model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
        is poi.TerminationStatusCode.OPTIMAL
    )
    # the equality is row 0, the two inequalities follow it
    duals = [
        model.get_constraint_attribute(c, poi.ConstraintAttribute.Dual)
        for c in (eq, c1, c2)
    ]
    assert duals == [0.0, 1.0, 2.0]


def test_read_parquet_solution_flips_a_maximisation_back(tmp_path):
    model, _, (eq, c1, c2) = _mixed_model(poi.ObjectiveSense.Maximize)
    stem = model.write_parquet(tmp_path / "model", layout="monolithic")
    _solution_beside(stem, sense=-1.0)

    model.read_parquet_solution(stem)

    # the file holds the minimisation objective; the model reports its own sense
    assert model.get_model_attribute(poi.ModelAttribute.ObjectiveValue) == 12.5
    duals = [
        model.get_constraint_attribute(c, poi.ConstraintAttribute.Dual)
        for c in (eq, c1, c2)
    ]
    assert duals == [-0.0, -1.0, -2.0]


# Test for unstructured models require the pipstools dependency
pipstools = pytest.importorskip("pipstools", reason="pipstools derives the structure")

# PyOptInterface names the variables here "x(17)", so this captures the index;
# "cap" carries none and stays in the root
INDEXED = r"\((\d+)\)$"

ANNOTATIONS = {
    "regex": {"method": "regex", "regex": INDEXED},
    "hypergraph": {"method": "hypergraph"},
}

N_TIME, N_BLOCKS = 12, 4


def _unstructured_model():
    """``cap`` shared by every period, ``x(t)`` per period, and no block stated."""
    demand = [1.0 + t / (N_TIME - 1) for t in range(N_TIME)]
    model = backend.Model()
    cap = model.add_variable(lb=0.0, name="cap")
    x = [model.add_variable(lb=0.0, name=f"x({t})") for t in range(N_TIME)]
    for t in range(N_TIME):
        model.add_linear_constraint(x[t], poi.Geq, demand[t], name=f"demand({t})")
        model.add_linear_constraint(x[t] - cap, poi.Leq, 0.0, name=f"cap({t})")
    model.set_objective(
        cap + 0.1 * poi.quicksum(x), poi.ObjectiveSense.Minimize
    )
    # x(t) = demand[t] and cap = max(demand), which needs no second solver
    model.hand_computed = max(demand) + 0.1 * sum(demand)
    return model, cap, x


def _skip_unless(method):
    if method == "hypergraph":
        pytest.importorskip("mtkahypar")


@pytest.mark.parametrize("method", sorted(ANNOTATIONS))
def test_annotate_fills_in_a_structure_that_was_never_stated(method):
    _skip_unless(method)
    model, _cap, _x = _unstructured_model()
    assert set(model._block) == {0}, "the model must start with no structure"

    leaves = model.annotate(N_BLOCKS, **ANNOTATIONS[method])

    assert leaves == N_BLOCKS
    assert max(model._block) == N_BLOCKS
    assert sorted(set(model._block)) == list(range(min(model._block), N_BLOCKS + 1))


def test_the_derived_blocks_are_readable_as_a_variable_attribute():
    model, cap, x = _unstructured_model()
    model.annotate(N_BLOCKS, **ANNOTATIONS["regex"])

    block = poi.VariableAttribute.Block
    assert model.get_variable_attribute(cap, block) == 0
    assert all(model.get_variable_attribute(v, block) != 0 for v in x)


def test_regex_without_named_variables_is_refused():
    model = backend.Model()
    a = model.add_variable(lb=0.0)
    b = model.add_variable(lb=0.0)
    model.add_linear_constraint(a + b, poi.Geq, 1.0)
    model.set_objective(a + b, poi.ObjectiveSense.Minimize)

    with pytest.raises(ValueError, match="none of the variables"):
        model.annotate(2, method="regex", regex=INDEXED)


@pytest.mark.parametrize("method", sorted(ANNOTATIONS))
def test_a_derived_structure_carries_into_the_written_files(tmp_path, method):
    _skip_unless(method)
    model, _cap, _x = _unstructured_model()
    model.annotate(N_BLOCKS, **ANNOTATIONS[method])

    manifest = pipsipmpp.read_manifest(model.write_parquet(tmp_path / method))

    # the root counts alongside the leaves
    assert manifest["n_blocks"] == N_BLOCKS + 1
    assert sum(block["n"] for block in manifest["blocks"]) == N_TIME + 1


@pytest.mark.parametrize("method", sorted(ANNOTATIONS))
def test_a_derived_structure_solves_and_survives_a_parquet_roundtrip(tmp_path, method):
    """Build here, solve there, read the answer back: the three steps apart."""
    _skip_unless(method)
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 1:
        pytest.skip("this test drives MPI itself; run it on a single rank")

    # in memory
    memory, memory_cap, memory_x = _unstructured_model()
    memory.annotate(N_BLOCKS, **ANNOTATIONS[method])
    memory.optimize()
    assert (
        memory.get_model_attribute(poi.ModelAttribute.TerminationStatus)
        == poi.TerminationStatusCode.OPTIMAL
    )
    objective = memory.get_model_attribute(poi.ModelAttribute.ObjectiveValue)
    assert objective == pytest.approx(memory.hand_computed, abs=1e-5)

    # step one: annotate and write, with no solver involved
    files, files_cap, files_x = _unstructured_model()
    files.annotate(N_BLOCKS, **ANNOTATIONS[method])
    stem = files.write_parquet(tmp_path / f"{method}-model", layout="distributed")
    # step two: solve from the files alone
    pipsipmpp.solve_dataset(stem, comm, write_solution=True)
    # step three: read it back onto the model, which needs no solver either
    files.read_parquet_solution(stem)

    assert files.get_model_attribute(
        poi.ModelAttribute.ObjectiveValue
    ) == pytest.approx(objective, abs=1e-6)
    assert files.get_value(files_cap) == pytest.approx(
        memory.get_value(memory_cap), abs=1e-6
    )
    for got, want in zip(files_x, memory_x):
        assert files.get_value(got) == pytest.approx(memory.get_value(want), abs=1e-6)
