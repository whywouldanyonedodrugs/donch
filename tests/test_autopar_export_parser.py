import unittest
from datetime import datetime, timezone

from tools.export_autopar_daily import (
    build_schema_diagnostics,
    enrich_decision_rows,
    parse_meta_decision_lines,
)


class TestAutoparExportParser(unittest.TestCase):
    def test_parse_meta_decision_line(self) -> None:
        lines = [
            "2026-02-10 10:04:15 [INFO] bundle=abc META_DECISION "
            "bundle=abc symbol=NEWTUSDT decision_ts=2026-02-10T09:50:00+00:00 "
            "schema_ok=True p_cal=0.3676 pstar=0.4200 pstar_scope=None "
            "risk_on_1=0.0 risk_on=0.0 scope_val=None scope_src=None scope_ok=True "
            "meta_ok=False strat_ok=False reason=below_pstar err=None"
        ]
        rows, raw = parse_meta_decision_lines(lines)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(raw), 1)
        r = rows[0]
        self.assertEqual(r["symbol"], "NEWTUSDT")
        self.assertEqual(r["decision"], "skipped")
        self.assertEqual(r["bundle"], "abc")
        self.assertAlmostEqual(r["p_cal"], 0.3676, places=8)
        self.assertAlmostEqual(r["pstar"], 0.42, places=8)
        self.assertEqual(r["pstar_scope"], "")
        self.assertEqual(r["risk_on"], 0)
        self.assertEqual(r["reason"], "below_pstar")
        self.assertEqual(r["reason_raw"], "below_pstar")
        self.assertEqual(r["reason_canonical"], "meta_prob")
        self.assertEqual(r["err"], "")

    def test_parse_err_with_spaces(self) -> None:
        lines = [
            "2026-02-10 10:04:30 [INFO] bundle=abc META_DECISION "
            "bundle=abc symbol=NIGHTUSDT decision_ts=2026-02-10T09:50:00+00:00 "
            "schema_ok=False p_cal=nan pstar=0.4200 pstar_scope=None "
            "risk_on_1=0.0 risk_on=0.0 scope_val=None scope_src=None scope_ok=True "
            "meta_ok=False strat_ok=False reason=schema_fail "
            "err=missing_required:['days_since_prev_break', 'S6_fresh_x_compress']"
        ]
        rows, _ = parse_meta_decision_lines(lines)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "skipped")
        self.assertIsNone(rows[0]["p_cal"])
        self.assertEqual(rows[0]["reason"], "schema_fail")
        self.assertEqual(
            rows[0]["err"],
            "missing_required:['days_since_prev_break', 'S6_fresh_x_compress']",
        )

    def test_dedup_latest_wins(self) -> None:
        k = (
            "symbol=BTCUSDT decision_ts=2026-02-10T10:00:00+00:00 "
            "schema_ok=True p_cal=0.50 pstar=0.42 pstar_scope=None "
            "risk_on_1=1.0 risk_on=1.0 scope_val=None scope_src=None scope_ok=True "
            "meta_ok=True strat_ok=False reason=rule_block err=None"
        )
        lines = [
            f"2026-02-10 10:00:01 [INFO] bundle=x META_DECISION bundle=x {k}",
            f"2026-02-10 10:00:02 [INFO] bundle=x META_DECISION bundle=x "
            "symbol=BTCUSDT decision_ts=2026-02-10T10:00:00+00:00 "
            "schema_ok=True p_cal=0.50 pstar=0.42 pstar_scope=None "
            "risk_on_1=1.0 risk_on=1.0 scope_val=None scope_src=None scope_ok=True "
            "meta_ok=True strat_ok=True reason=ok err=None",
        ]
        rows, _ = parse_meta_decision_lines(lines)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "taken")
        self.assertEqual(rows[0]["reason"], "ok")
        self.assertEqual(rows[0]["reason_canonical"], "ok")

    def test_schema_diagnostics_non_warmup_detection(self) -> None:
        lines = [
            "2026-02-10 10:04:30 [INFO] bundle=abc META_DECISION "
            "bundle=abc symbol=AUSDT decision_ts=2026-02-10T09:50:00+00:00 "
            "schema_ok=False p_cal=nan pstar=0.4200 pstar_scope=None "
            "risk_on_1=0.0 risk_on=0.0 scope_val=None scope_src=None scope_ok=True "
            "meta_ok=False strat_ok=False reason=schema_fail "
            "err=missing_required:['days_since_prev_break', 'S6_fresh_x_compress']",
            "2026-02-10 10:04:31 [INFO] bundle=abc META_DECISION "
            "bundle=abc symbol=BUSDT decision_ts=2026-02-10T09:50:00+00:00 "
            "schema_ok=False p_cal=nan pstar=0.4200 pstar_scope=None "
            "risk_on_1=0.0 risk_on=0.0 scope_val=None scope_src=None scope_ok=True "
            "meta_ok=False strat_ok=False reason=schema_fail "
            "err=missing_required:['bad_live_field']",
            "2026-02-10 10:04:32 [INFO] bundle=abc META_DECISION "
            "bundle=abc symbol=BUSDT decision_ts=2026-02-10T09:55:00+00:00 "
            "schema_ok=False p_cal=nan pstar=0.4200 pstar_scope=None "
            "risk_on_1=0.0 risk_on=0.0 scope_val=None scope_src=None scope_ok=True "
            "meta_ok=False strat_ok=False reason=schema_fail "
            "err=missing_required:['bad_live_field']",
            "2026-02-10 10:04:33 [INFO] bundle=abc META_DECISION "
            "bundle=abc symbol=CUSDT decision_ts=2026-02-10T09:50:00+00:00 "
            "schema_ok=True p_cal=0.45 pstar=0.4200 pstar_scope=None "
            "risk_on_1=1.0 risk_on=1.0 scope_val=None scope_src=None scope_ok=True "
            "meta_ok=True strat_ok=False reason=rule_block err=None",
        ]
        rows, _ = parse_meta_decision_lines(lines)
        enrich_decision_rows(
            rows,
            symbol_meta={
                "AUSDT": {
                    "listed_at_dt": datetime(2026, 2, 1, tzinfo=timezone.utc),
                    "symbol_listed_at_utc": "2026-02-01T00:00:00+00:00",
                    "metadata_source": "exchange_instrument_metadata",
                },
                "BUSDT": {
                    "listed_at_dt": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "symbol_listed_at_utc": "2025-01-01T00:00:00+00:00",
                    "metadata_source": "exchange_instrument_metadata",
                },
                "CUSDT": {
                    "listed_at_dt": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "symbol_listed_at_utc": "2024-01-01T00:00:00+00:00",
                    "metadata_source": "exchange_instrument_metadata",
                },
            },
            warmup_max_age_days=45,
        )
        diag = build_schema_diagnostics(rows)
        self.assertEqual(diag["decisions_count"], 4)
        self.assertEqual(diag["schema_fail_count"], 3)
        self.assertAlmostEqual(diag["schema_fail_rate"], 0.75, places=8)
        self.assertEqual(diag["warmup_only_schema_fail_count"], 1)
        self.assertEqual(diag["non_warmup_schema_fail_count"], 2)
        self.assertEqual(diag["top_non_warmup_missing_fields"][0]["field"], "bad_live_field")
        self.assertEqual(diag["top_non_warmup_missing_fields"][0]["count"], 2)
        self.assertEqual(diag["schema_fail_class_counts"]["warmup_expected"], 1)
        self.assertEqual(diag["schema_fail_class_counts"]["integration_defect"], 2)
        self.assertEqual(diag["schema_fail_class_counts"]["unknown"], 0)
        self.assertFalse(diag["incident_recommended"])


if __name__ == "__main__":
    unittest.main()
