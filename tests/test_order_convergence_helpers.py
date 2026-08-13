"""Tests for the pure helpers behind the order-convergence table.

These decide what the Milestone 3 acceptance-gate table asserts: `_pin_key` licenses
printing a shift as `0.00*` ("fixed by the realization, not measured") rather than as
convergence, and `_runs`/`_span` license "identically so at all seven stepped-down orders"
by collapsing repeats into a range. A regression in either produces a table that reads as
evidence while being wrong, and the only other way to catch it is to re-run a 40-second
sweep and compare by eye. The rest of the study is physics and stays untested here.
"""

import math

import pytest

from baseline.studies.order_convergence import (
    LayoutError,
    _by_sector,
    _pin_key,
    _runs,
    _sectors,
    _span,
)


def sectors(hole, particle, hole_residual=1e-15, particle_residual=1e-15):
    """Build the per-sector mapping `_by_sector` produces, without running a calculation."""
    return {
        "hole": {"conserved": hole, "residual": hole_residual},
        "particle": {"conserved": particle, "residual": particle_residual},
    }


class TestPinKey:
    def test_same_conserved_orders_are_pinned(self):
        assert _pin_key(sectors(18, 20)) == _pin_key(sectors(18, 20))

    def test_a_sector_gaining_is_not_pinned(self):
        assert _pin_key(sectors(18, 20)) != _pin_key(sectors(18, 22))

    def test_nan_residuals_do_not_switch_the_marking_off(self):
        """The reason the key exists rather than comparing the dicts.

        `residual` is `nan` whenever `calculate_errors` is off, and `nan != nan` makes plain
        dict equality false on two identical realizations - silently disabling every mark.
        """
        # Distinct `nan` objects, as separate `float(...)` calls in `_by_sector` produce.
        # Reusing one object would make the dicts compare equal by identity and the test
        # would pass without exercising anything.
        a = sectors(18, 20, float("nan"), float("nan"))
        b = sectors(18, 20, float("nan"), float("nan"))

        assert a != b  # what the marking used to rest on
        assert _pin_key(a) == _pin_key(b)

    def test_residual_does_not_enter_the_key(self):
        """Two orders that conserve the same realize the same, whatever the residual reads.

        Keeping the residual out also keeps the mark off a float comparison, which CLAUDE.md
        notes moves with the BLAS summation order.
        """
        assert _pin_key(sectors(18, 20, 1e-15)) == _pin_key(sectors(18, 20, 2e-15))


class TestRuns:
    def test_a_contiguous_sweep_is_one_run(self):
        assert _runs([19, 21, 23], 2) == [[19, 21, 23]]

    def test_a_gap_starts_a_new_run(self):
        """The property the range collapsing rests on.

        A skipped order is one where no probe ran, or where the sector recovered; printing
        `K=19 to 25` across it would assert a measurement that was never taken.
        """
        assert _runs([19, 25, 27], 2) == [[19], [25, 27]]

    def test_the_stride_is_honoured_rather_than_assumed(self):
        assert _runs([1, 2, 3], 1) == [[1, 2, 3]]
        assert _runs([1, 2, 3], 2) == [[1], [2], [3]]

    def test_single_and_empty(self):
        assert _runs([19], 2) == [[19]]
        assert _runs([], 2) == []


class TestSpan:
    def test_one_order_is_not_rendered_as_a_range(self):
        assert _span([19], 2) == "K=19"

    def test_a_run_is_rendered_as_its_endpoints(self):
        assert _span([19, 21, 23], 2) == "K=19 to 23"

    def test_disjoint_runs_are_listed_separately(self):
        assert _span([19, 25, 27], 2) == "K=19, K=25 to 27"

    def test_empty(self):
        assert _span([], 2) == ""


class TestSectors:
    def test_fields_are_joined_in_the_diagnostics_order(self):
        assert _sectors(sectors(18, 20), "conserved") == "18/20"

    def test_the_format_is_applied(self):
        assert _sectors(sectors(1, 1, 1.1e-14, 3.3e4), "residual", "{:.2e}") == "1.10e-14/3.30e+04"


class TestLayoutGuard:
    """The guard that keeps a structural fault out of the probe loop's result reporting.

    `_psd_probe` failures are caught and printed as findings about the PSD gate. A renamed
    sector or a renamed record field is a bug in this study, and without a typed error raised
    ahead of every field read it would surface as `KeyError: 'errors'` under a gate heading.
    """

    def record(self, **overrides):
        base = {
            "nmom_conserved_achieved": 6,
            "nmom_conserved_requested": 8,
            "moments_supplied": 8,
            "errors": None,
        }
        base.update(overrides)
        return base

    def test_the_expected_layout_is_accepted(self):
        by_sector = _by_sector({"hole": self.record(), "particle": self.record()})

        assert by_sector["hole"]["conserved"] == 6
        assert by_sector["hole"]["shortfall"] == 2

    def test_an_unexpected_sector_is_rejected(self):
        with pytest.raises(LayoutError, match="laid out for sectors"):
            _by_sector({"hole": self.record(), "particle": self.record(), "spin": self.record()})

    def test_reordered_sectors_are_rejected(self):
        """Column order is positional, so particle-first would silently mislabel every row."""
        with pytest.raises(LayoutError, match="laid out for sectors"):
            _by_sector({"particle": self.record(), "hole": self.record()})

    @pytest.mark.parametrize(
        "missing",
        ["nmom_conserved_achieved", "nmom_conserved_requested", "moments_supplied", "errors"],
    )
    def test_a_renamed_record_field_is_rejected_before_it_is_read(self, missing):
        """The reason the check runs ahead of the loop rather than after it."""
        record = self.record()
        del record[missing]

        with pytest.raises(LayoutError, match=missing):
            _by_sector({"hole": record, "particle": self.record()})

    def test_errors_none_becomes_nan_rather_than_raising(self):
        """`calculate_errors=False` is a supported configuration, not a layout fault."""
        by_sector = _by_sector({"hole": self.record(), "particle": self.record()})

        assert math.isnan(by_sector["hole"]["residual"])
