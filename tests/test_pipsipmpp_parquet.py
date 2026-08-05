"""Writing an annotated model out as parquet for a later or remote solve."""

import pytest

import pyoptinterface as poi

pipsipmpppy = pytest.importorskip("pipsipmpppy")
pytest.importorskip("pyarrow")

from pyoptinterface import pipsipmpp  # noqa: E402


def _two_block_model(sense=poi.ObjectiveSense.Minimize):
    """Root variable ``y`` coupling two single-variable leaf blocks."""
    model = pipsipmpp.Model()
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

    manifest = pipsipmpppy.read_manifest(stem)
    assert manifest["layout"] == layout
    assert manifest["n_blocks"] == 3  # the root counts alongside the two leaves
    assert [block["n"] for block in manifest["blocks"]] == [1, 1, 1]


def test_both_layouts_describe_the_same_problem(tmp_path):
    whole = pipsipmpppy.read_manifest(
        _two_block_model().write_parquet(tmp_path / "whole")
    )
    split = pipsipmpppy.read_manifest(
        _two_block_model().write_parquet(tmp_path / "split", layout="distributed")
    )

    assert (whole["n_rows"], whole["n_cols"]) == (split["n_rows"], split["n_cols"])
    keys = ("n", "my", "mz", "myl", "mzl")
    assert [[b[k] for k in keys] for b in whole["blocks"]] == [
        [b[k] for k in keys] for b in split["blocks"]
    ]


def test_rows_and_columns_carry_their_model_names(tmp_path):
    stem = _two_block_model().write_parquet(tmp_path / "named")
    names = pipsipmpppy.read_names(stem)

    assert sorted(names["cols"]) == ["x1", "x2", "y"]
    assert sorted(names["rows"]) == ["cap1", "cap2", "demand1", "demand2"]


def test_names_can_be_left_out(tmp_path):
    stem = _two_block_model().write_parquet(tmp_path / "plain", names=False)
    assert pipsipmpppy.read_names(stem) == {}


def test_a_maximisation_records_its_sense(tmp_path):
    model = _two_block_model(poi.ObjectiveSense.Maximize)
    stem = model.write_parquet(tmp_path / "maxi")
    # the problem holds minimisation costs; objcoef says how the model stated them
    assert pipsipmpppy.read_manifest(stem)["objcoef"] == -1.0


def test_unknown_layout_is_reported(tmp_path):
    with pytest.raises(ValueError, match="unknown layout"):
        _two_block_model().write_parquet(tmp_path / "nope", layout="blockwise")
