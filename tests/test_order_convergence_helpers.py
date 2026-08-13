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
    _classify_residuals,
    _pin_key,
    _residual_report,
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


class TestClassifyResiduals:
    """The split that decides whether a probe verdict blames the loosening.

    Reporting a pre-existing blow-up as a cost of loosening voids orders another sector
    genuinely bought, which is what an earlier version of this did on lithium-hydride.
    """

    def test_healthy_everywhere_says_nothing(self):
        caused, already, repaired, unknown = _classify_residuals(sectors(6, 8), sectors(8, 8))

        assert (caused, already, repaired, unknown) == ({}, {}, {}, set())

    def test_a_blow_up_the_loosening_created_is_its_fault(self):
        before = sectors(6, 8, 1e-15, 1e-15)
        reached = sectors(8, 8, 1e3, 1e-15)

        caused, already, repaired, unknown = _classify_residuals(before, reached)

        assert caused == {"hole": 1e3}
        assert already == {} and repaired == {} and unknown == set()

    def test_a_blow_up_that_predates_it_is_not(self):
        """Lithium-hydride at nmom_max=19: the particle is broken before and after."""
        before = sectors(6, 20, 3.6e-15, 3.3e4)
        reached = sectors(8, 20, 2.2e-13, 3.3e4)

        caused, already, repaired, unknown = _classify_residuals(before, reached)

        assert already == {"particle": 3.3e4}
        # And the hole, which actually gained, is not implicated.
        assert caused == {} and repaired == {} and unknown == set()

    def test_a_missing_residual_is_neither(self):
        """`nan` is what `_by_sector` stores when errors are off.

        Every comparison against it is False, so without an explicit branch the sector lands
        in `already` and the verdict asserts the blow-up predates a loosening it may not.
        """
        before = sectors(6, 8, float("nan"), 1e-15)
        reached = sectors(8, 8, 1e3, 1e-15)

        caused, already, repaired, unknown = _classify_residuals(before, reached)

        assert unknown == {"hole"}
        assert caused == {} and already == {} and repaired == {}


def rows_with(*per_order):
    """Build the row list `_residual_report` reads.

    Each entry is `(order, hole_residual, particle_residual)`, optionally followed by the
    two conserved orders - the grouping key is `(sector, conserved)`, so a fixture that
    leaves conserved fixed cannot exercise it.
    """
    rows = []
    for order, hole, particle, *conserved in per_order:
        h, pt = conserved or (1, 1)
        rows.append({"order": order, "by_sector": sectors(h, pt, hole, particle), "surplus": {}})
    return rows


class TestResidualReport:
    """The footer that reports realizations which do not reproduce their moments.

    This is the study's only claim that a calculation is unusable, and the ROADMAP quotes
    its realization count, so a miscount is a misreported finding rather than a cosmetic bug.
    """

    def test_a_healthy_sweep_reports_nothing(self):
        assert _residual_report(rows_with((19, 1e-15, 1e-14), (21, 1e-15, 1e-14)), 2) == []

    def test_a_pinned_blow_up_is_one_realization_however_many_orders_repeat_it(self):
        """Once a sector pins, every higher order repeats the same failed realization."""
        rows = rows_with((19, 1e-15, 3e4), (21, 1e-15, 3e4), (23, 1e-15, 3e4))

        (line,) = _residual_report(rows, 2)

        assert "1 distinct realization(s)" in line
        assert "K=19 to 23" in line

    def test_recovering_and_blowing_up_again_is_two(self):
        """The case the count used to get wrong by keying on the sector alone.

        Same sector, same conserved order, but a healthy order in between: two separate
        failed realizations, not one run spanning an order that was fine.
        """
        rows = rows_with((19, 1e-15, 3e4), (21, 1e-15, 1e-14), (23, 1e-15, 3e4), (25, 1e-15, 3e4))

        (line,) = _residual_report(rows, 2)

        assert "2 distinct realization(s)" in line
        assert "K=19, K=23 to 25" in line

    def test_an_unmeasured_residual_is_reported_not_dropped(self):
        """`nan > RESIDUAL_MAX` is False, so filtering would read as a clean sweep."""
        (line,) = _residual_report(rows_with((19, float("nan"), 1e-14)), 2)

        assert "no residual was calculated for hole" in line
        assert "absence of evidence" in line

    def test_unmeasured_and_blown_are_both_reported(self):
        lines = _residual_report(rows_with((19, float("nan"), 3e4)), 2)

        assert len(lines) == 2
        assert "no residual was calculated" in lines[0]
        assert "1 distinct realization(s)" in lines[1]

    def test_the_same_sector_at_two_conserved_orders_is_two_realizations(self):
        """The `(sector, conserved)` key, which a fixed-conserved fixture cannot reach.

        Consecutive orders, so contiguity does not separate them - only the conserved order
        does, and that is what makes them two realizations rather than one run.
        """
        rows = rows_with((19, 1e-15, 3e4, 1, 20), (21, 1e-15, 3e4, 1, 22))

        (line,) = _residual_report(rows, 2)

        assert "2 distinct realization(s)" in line
        assert "particle at 20 conserved over K=19" in line
        assert "particle at 22 conserved over K=21" in line

    def test_the_worst_residual_is_the_one_named(self):
        rows = rows_with((19, 1e-15, 3e4), (21, 5e6, 3e4))

        (line,) = _residual_report(rows, 2)

        assert "reaches 5.00e+06 in the hole sector at K=21" in line


class TestClassifyResidualsRepair:
    """Loosening can also fix a blown residual, which is the outcome easiest to drop."""

    def test_a_repaired_sector_is_reported(self):
        before = sectors(6, 20, 1e-15, 3.3e4)
        reached = sectors(6, 20, 1e-15, 1e-15)

        caused, already, repaired, unknown = _classify_residuals(before, reached)

        assert repaired == {"particle": 1e-15}
        assert caused == {} and already == {} and unknown == set()


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
