import sqlite3

import pytest

import app.store as store
from app.protocol_study import calculate_reduction


def test_scan_time_reduction():
    assert calculate_reduction(240, 180) == 25.0
    assert calculate_reduction(100, 65) == 35.0
    with pytest.raises(ValueError):
        calculate_reduction(100, 100)


def test_protocol_jobs_cannot_enter_return_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test.sqlite3"))
    store.initialize()
    with store.connect() as db:
        center = db.execute(
            "INSERT INTO centers(name,slug,source_ip,source_aet,return_ip,return_aet,return_port) VALUES(?,?,?,?,?,?,?)",
            ("Test", "test", "127.0.0.1", "TEST_AE", "127.0.0.1", "RETURN_AE", 11112),
        ).lastrowid
        protocol = db.execute(
            "INSERT INTO protocol_studies(title,anatomy,sequence,standard_series_id,fast_series_id,standard_seconds,fast_seconds,reduction_percent) VALUES(?,?,?,?,?,?,?,?)",
            ("Pilot", "Knee", "PD", "standard", "fast", 240, 180, 25),
        ).lastrowid
        job = db.execute(
            "INSERT INTO comparison_jobs(source_series_id,profile,strength,protocol_study_id,output_series_id,status) VALUES(?,?,?,?,?,?)",
            ("fast", "EDGE_DENOISE_MILD", "MILD", protocol, "derived", "ready"),
        ).lastrowid
        with pytest.raises(sqlite3.IntegrityError, match="Protocol Study results cannot be returned"):
            db.execute(
                "INSERT INTO return_jobs(comparison_job_id,center_id,output_series_id) VALUES(?,?,?)",
                (job, center, "derived"),
            )
        generic_job = db.execute(
            "INSERT INTO comparison_jobs(source_series_id,profile,strength,output_series_id,status) VALUES(?,?,?,?,?)",
            ("fast", "EDGE_DENOISE_MILD", "MILD", "derived-generic", "ready"),
        ).lastrowid
        with pytest.raises(sqlite3.IntegrityError, match="Protocol Study results cannot be returned"):
            db.execute(
                "INSERT INTO return_jobs(comparison_job_id,center_id,output_series_id) VALUES(?,?,?)",
                (generic_job, center, "derived-generic"),
            )
