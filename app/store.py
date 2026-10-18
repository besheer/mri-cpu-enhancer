from __future__ import annotations

import os
import re
import sqlite3

DB_PATH = os.getenv("APP_DB", "/var/lib/enhancer/enhancer.sqlite3")


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def initialize() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS centers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              slug TEXT NOT NULL UNIQUE,
              source_ip TEXT NOT NULL,
              source_aet TEXT NOT NULL UNIQUE,
              return_ip TEXT NOT NULL,
              return_aet TEXT NOT NULL,
              return_port INTEGER NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS processed_series (
              series_id TEXT PRIMARY KEY,
              center_id INTEGER,
              status TEXT NOT NULL,
              error TEXT,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS pending_senders (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source_ip TEXT NOT NULL,
              source_aet TEXT NOT NULL,
              called_aet TEXT NOT NULL DEFAULT '',
              attempts INTEGER NOT NULL DEFAULT 1,
              status TEXT NOT NULL DEFAULT 'pending',
              first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(source_ip, source_aet, called_aet)
            );
            CREATE TABLE IF NOT EXISTS comparison_jobs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source_series_id TEXT NOT NULL,
              profile TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'queued',
              output_series_id TEXT,
              metrics_json TEXT,
              error TEXT,
              approved INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS processing_tools (
              profile TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,
              available INTEGER NOT NULL DEFAULT 1,
              auto_eligible INTEGER NOT NULL DEFAULT 1,
              notes TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS return_jobs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              comparison_job_id INTEGER NOT NULL,
              center_id INTEGER NOT NULL,
              output_series_id TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'queued',
              attempts INTEGER NOT NULL DEFAULT 0,
              error TEXT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(comparison_job_id),
              FOREIGN KEY(center_id) REFERENCES centers(id)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              actor TEXT NOT NULL DEFAULT 'system',
              action TEXT NOT NULL,
              object_type TEXT NOT NULL,
              object_id TEXT NOT NULL,
              detail TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS training_datasets (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              anatomy TEXT NOT NULL,
              sequence TEXT NOT NULL,
              data_mode TEXT NOT NULL,
              path TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'REGISTERED',
              notes TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS model_registry (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              version TEXT NOT NULL,
              format TEXT NOT NULL DEFAULT 'ONNX',
              path TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'STAGED',
              validation_json TEXT NOT NULL DEFAULT '{}',
              active INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(name,version)
            );
            CREATE TABLE IF NOT EXISTS inbound_series (
              series_id TEXT PRIMARY KEY,
              center_id INTEGER,
              source_aet TEXT NOT NULL DEFAULT '',
              study_id TEXT NOT NULL DEFAULT '',
              description TEXT NOT NULL DEFAULT '',
              instances INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL DEFAULT 'RECEIVED',
              job_id INTEGER,
              error TEXT,
              received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(center_id) REFERENCES centers(id)
            );
            CREATE TABLE IF NOT EXISTS system_settings (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS synthetic_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              test_type TEXT NOT NULL,
              severity TEXT NOT NULL,
              seed INTEGER NOT NULL,
              study_uid TEXT NOT NULL,
              clean_series_id TEXT NOT NULL,
              corrupted_series_id TEXT NOT NULL,
              input_metrics_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS synthetic_results (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id INTEGER NOT NULL,
              profile TEXT NOT NULL,
              strength TEXT NOT NULL,
              output_series_id TEXT NOT NULL,
              difference_series_id TEXT NOT NULL DEFAULT '',
              metrics_json TEXT NOT NULL DEFAULT '{}',
              decision TEXT NOT NULL DEFAULT '',
              rank_score REAL NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(run_id) REFERENCES synthetic_runs(id)
            );
            CREATE TABLE IF NOT EXISTS protocol_studies (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT NOT NULL,
              center_id INTEGER,
              anatomy TEXT NOT NULL,
              sequence TEXT NOT NULL,
              scanner_model TEXT NOT NULL DEFAULT '',
              coil TEXT NOT NULL DEFAULT '',
              standard_series_id TEXT NOT NULL,
              fast_series_id TEXT NOT NULL,
              study_instance_uid TEXT NOT NULL DEFAULT '',
              standard_seconds REAL NOT NULL,
              fast_seconds REAL NOT NULL,
              reduction_percent REAL NOT NULL,
              protocol_notes TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL DEFAULT 'DRAFT',
              archived INTEGER NOT NULL DEFAULT 0,
              created_by TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(center_id) REFERENCES centers(id)
            );
            CREATE TABLE IF NOT EXISTS protocol_reviews (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              protocol_study_id INTEGER NOT NULL,
              reviewer TEXT NOT NULL,
              image_quality INTEGER NOT NULL,
              diagnostic_confidence INTEGER NOT NULL,
              noise INTEGER NOT NULL,
              sharpness INTEGER NOT NULL,
              artifacts INTEGER NOT NULL,
              decision TEXT NOT NULL,
              notes TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(protocol_study_id) REFERENCES protocol_studies(id)
            );
            """
        )
        columns = {row["name"] for row in db.execute("PRAGMA table_info(centers)")}
        migrations = {
            "public_ip": "TEXT NOT NULL DEFAULT ''",
            "public_ip_static": "INTEGER NOT NULL DEFAULT 1",
            "pacs_internal_ip": "TEXT NOT NULL DEFAULT ''",
            "device_manufacturer": "TEXT NOT NULL DEFAULT ''",
            "device_model": "TEXT NOT NULL DEFAULT ''",
            "field_strength": "TEXT NOT NULL DEFAULT '0.3'",
            "anatomy": "TEXT NOT NULL DEFAULT ''",
            "sequences": "TEXT NOT NULL DEFAULT ''",
            "notes": "TEXT NOT NULL DEFAULT ''",
            "processing_profile": "TEXT NOT NULL DEFAULT 'BASE_ONLY'",
            "network_mode": "TEXT NOT NULL DEFAULT 'UNKNOWN'",
            "return_policy": "TEXT NOT NULL DEFAULT 'MANUAL'",
            "vpn_cidr": "TEXT NOT NULL DEFAULT ''",
            "max_return_attempts": "INTEGER NOT NULL DEFAULT 3",
            "last_echo_status": "TEXT NOT NULL DEFAULT 'NOT_TESTED'",
            "last_echo_at": "TEXT",
            "auto_enhance_enabled": "INTEGER NOT NULL DEFAULT 0",
            "auto_profile": "TEXT NOT NULL DEFAULT 'EDGE_DENOISE_MILD'",
            "auto_strength": "TEXT NOT NULL DEFAULT 'MILD'",
        }
        for name, declaration in migrations.items():
            if name not in columns:
                db.execute(f"ALTER TABLE centers ADD COLUMN {name} {declaration}")
        job_columns = {row["name"] for row in db.execute("PRAGMA table_info(comparison_jobs)")}
        job_migrations = {
            "strength": "TEXT NOT NULL DEFAULT 'MEDIUM'",
            "qc_status": "TEXT NOT NULL DEFAULT 'PENDING'",
            "qc_reasons": "TEXT NOT NULL DEFAULT ''",
            "decision": "TEXT NOT NULL DEFAULT ''",
            "review_notes": "TEXT NOT NULL DEFAULT ''",
            "reviewed_at": "TEXT",
            "total_instances": "INTEGER NOT NULL DEFAULT 0",
            "processed_instances": "INTEGER NOT NULL DEFAULT 0",
            "archived": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, declaration in job_migrations.items():
            if name not in job_columns:
                db.execute(f"ALTER TABLE comparison_jobs ADD COLUMN {name} {declaration}")
        if "protocol_study_id" not in job_columns:
            db.execute("ALTER TABLE comparison_jobs ADD COLUMN protocol_study_id INTEGER")
        for table in ("return_jobs", "inbound_series"):
            table_columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
            if "archived" not in table_columns:
                db.execute(f"ALTER TABLE {table} ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        tool_columns = {row["name"] for row in db.execute("PRAGMA table_info(processing_tools)")}
        if "auto_eligible" not in tool_columns:
            db.execute("ALTER TABLE processing_tools ADD COLUMN auto_eligible INTEGER NOT NULL DEFAULT 1")
        synthetic_columns = {row["name"] for row in db.execute("PRAGMA table_info(synthetic_results)")}
        for name, declaration in {
            "difference_series_id": "TEXT NOT NULL DEFAULT ''",
            "decision": "TEXT NOT NULL DEFAULT ''",
            "rank_score": "REAL NOT NULL DEFAULT 0",
        }.items():
            if name not in synthetic_columns:
                db.execute(f"ALTER TABLE synthetic_results ADD COLUMN {name} {declaration}")
        db.executemany("INSERT OR IGNORE INTO system_settings(key,value) VALUES(?,?)", [
            ("pause_intake", "0"), ("pause_enhancement", "0"), ("pause_return", "0"),
            ("maintenance_mode", "0"),
        ])
        db.execute("UPDATE comparison_jobs SET status='queued',error='Recovered after restart' WHERE status='processing'")
        tools = [
            ("BASE_MILD", "Baseline محافظ", 1, 1, "المرجع الافتراضي"),
            ("DENOISE_MEDIUM", "Denoise متوسط", 1, 1, "للضوضاء الواضحة"),
            ("EDGE_DENOISE_MILD", "Edge Denoise محافظ", 1, 1, "تقليل ضوضاء مع حماية الحواف"),
            ("N4_BIAS", "N4 Bias فقط", 1, 1, "لتفاوت الإضاءة على Volume"),
            ("N4_DENOISE", "N4 + Denoise", 1, 1, "Bias وضوضاء معًا"),
            ("N4_EDGE_DENOISE", "N4 + Edge Denoise", 1, 1, "Bias وضوضاء مع حماية الحواف"),
            ("CLAHE_VIEW", "CLAHE للعرض البحثي", 0, 1, "عرض بحثي فقط"),
            ("MONAI_MODEL", "MONAI Model", 0, 0, "غير متاح حتى رفع نموذج موثق"),
            ("DIPY_DWI", "DIPY DWI", 0, 0, "غير متاح حتى دعم DWI metadata"),
        ]
        db.executemany("INSERT OR IGNORE INTO processing_tools(profile,label,enabled,available,notes) VALUES(?,?,?,?,?)", tools)
        # Reference/display/unvalidated engines remain available for controlled
        # comparison but can never be selected by branch automation.
        db.execute("UPDATE processing_tools SET auto_eligible=0 WHERE profile IN ('BASE_MILD','CLAHE_VIEW','MONAI_MODEL','DIPY_DWI')")
        db.execute("UPDATE centers SET auto_enhance_enabled=0 WHERE auto_profile IN ('BASE_MILD','CLAHE_VIEW') OR auto_strength='STRONG'")
        # Protocol-acceleration experiments are comparison-only. This database
        # guard prevents a UI or worker mistake from placing them in Return Queue.
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS block_protocol_study_return
            BEFORE INSERT ON return_jobs
            WHEN EXISTS (
              SELECT 1 FROM comparison_jobs
              WHERE id=NEW.comparison_job_id AND protocol_study_id IS NOT NULL
            )
            OR EXISTS (
              SELECT 1 FROM comparison_jobs j
              JOIN protocol_studies p
                ON p.archived=0
               AND (p.standard_series_id=j.source_series_id OR p.fast_series_id=j.source_series_id)
              WHERE j.id=NEW.comparison_job_id
            )
            BEGIN
              SELECT RAISE(ABORT, 'Protocol Study results cannot be returned');
            END
        """)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "center"


def list_centers():
    with connect() as db:
        return db.execute("SELECT * FROM centers ORDER BY name").fetchall()


def get_center(center_id: int):
    with connect() as db:
        return db.execute("SELECT * FROM centers WHERE id=?", (center_id,)).fetchone()


def list_queue(limit: int = 100):
    with connect() as db:
        return db.execute(
            "SELECT p.*, c.name AS center_name FROM processed_series p "
            "LEFT JOIN centers c ON c.id=p.center_id "
            "ORDER BY p.updated_at DESC LIMIT ?", (limit,)
        ).fetchall()


def dashboard_counts() -> dict[str, int]:
    with connect() as db:
        return {
            "centers": db.execute("SELECT COUNT(*) FROM centers").fetchone()[0],
            "enabled": db.execute("SELECT COUNT(*) FROM centers WHERE enabled=1").fetchone()[0],
            "processing": db.execute("SELECT COUNT(*) FROM processed_series WHERE status='processing'").fetchone()[0],
            "failed": db.execute("SELECT COUNT(*) FROM processed_series WHERE status='failed'").fetchone()[0],
            "pending": db.execute("SELECT COUNT(*) FROM pending_senders WHERE status='pending'").fetchone()[0],
        }


def list_pending_senders():
    with connect() as db:
        return db.execute("SELECT * FROM pending_senders WHERE status='pending' ORDER BY last_seen DESC").fetchall()


def get_pending_sender(sender_id: int):
    with connect() as db:
        return db.execute("SELECT * FROM pending_senders WHERE id=?", (sender_id,)).fetchone()


def record_pending_sender(ip: str, aet: str, called_aet: str = "") -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO pending_senders(source_ip,source_aet,called_aet) VALUES(?,?,?) "
            "ON CONFLICT(source_ip,source_aet,called_aet) DO UPDATE SET "
            "attempts=pending_senders.attempts+1,status='pending',last_seen=CURRENT_TIMESTAMP",
            (ip, aet, called_aet),
        )


def enqueue_comparison(series_id: str, profile: str, strength: str = "MEDIUM") -> int:
    with connect() as db:
        existing = db.execute(
            "SELECT id FROM comparison_jobs WHERE source_series_id=? AND profile=? AND strength=? "
            "AND status IN ('queued','processing','ready') ORDER BY id DESC LIMIT 1",
            (series_id, profile, strength),
        ).fetchone()
        if existing:
            return int(existing["id"])
        cursor = db.execute(
            "INSERT INTO comparison_jobs(source_series_id,profile,strength) VALUES(?,?,?)",
            (series_id, profile, strength),
        )
        return int(cursor.lastrowid)


def list_comparison_jobs(limit: int = 100, status: str = "", profile: str = ""):
    with connect() as db:
        where, values = ["archived=0"], []
        if status:
            where.append("status=?")
            values.append(status)
        if profile:
            where.append("profile=?")
            values.append(profile)
        clause = " WHERE " + " AND ".join(where) if where else ""
        return db.execute(
            f"SELECT * FROM comparison_jobs{clause} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*values, limit),
        ).fetchall()


def get_comparison_job(job_id: int):
    with connect() as db:
        return db.execute("SELECT * FROM comparison_jobs WHERE id=?", (job_id,)).fetchone()


def list_processing_tools():
    with connect() as db:
        return db.execute("SELECT * FROM processing_tools ORDER BY available DESC, profile").fetchall()


def enabled_profile_names() -> set[str]:
    with connect() as db:
        return {row[0] for row in db.execute("SELECT profile FROM processing_tools WHERE enabled=1 AND available=1")}


def auto_eligible_profile_names() -> set[str]:
    with connect() as db:
        return {row[0] for row in db.execute(
            "SELECT profile FROM processing_tools WHERE enabled=1 AND available=1 AND auto_eligible=1"
        )}


def get_center_by_aet(aet: str):
    with connect() as db:
        return db.execute(
            "SELECT * FROM centers WHERE source_aet=? AND enabled=1", (aet,)
        ).fetchone()


def enqueue_return(comparison_job_id: int, center_id: int, output_series_id: str) -> int:
    with connect() as db:
        db.execute(
            "INSERT OR IGNORE INTO return_jobs(comparison_job_id,center_id,output_series_id) VALUES(?,?,?)",
            (comparison_job_id, center_id, output_series_id),
        )
        row = db.execute("SELECT id FROM return_jobs WHERE comparison_job_id=?", (comparison_job_id,)).fetchone()
        return int(row["id"])


def list_return_jobs(limit: int = 100):
    with connect() as db:
        return db.execute(
            "SELECT r.*,c.name AS center_name,c.return_aet,c.return_ip,c.return_port "
            "FROM return_jobs r JOIN centers c ON c.id=r.center_id WHERE r.archived=0 "
            "ORDER BY r.updated_at DESC,r.id DESC LIMIT ?", (limit,)
        ).fetchall()


def list_inbound_series(limit: int = 100):
    with connect() as db:
        return db.execute(
            "SELECT i.*,c.name AS center_name FROM inbound_series i "
            "LEFT JOIN centers c ON c.id=i.center_id WHERE i.archived=0 "
            "ORDER BY i.updated_at DESC LIMIT ?", (limit,)
        ).fetchall()


def pipeline_counts() -> dict[str, int]:
    with connect() as db:
        return {
            "received": db.execute("SELECT COUNT(*) FROM inbound_series WHERE archived=0 AND status='RECEIVED'").fetchone()[0],
            "queued": db.execute("SELECT COUNT(*) FROM comparison_jobs WHERE archived=0 AND status='queued'").fetchone()[0],
            "processing": db.execute("SELECT COUNT(*) FROM comparison_jobs WHERE archived=0 AND status='processing'").fetchone()[0],
            "ready": db.execute("SELECT COUNT(*) FROM comparison_jobs WHERE archived=0 AND status='ready'").fetchone()[0],
            "return_queued": db.execute("SELECT COUNT(*) FROM return_jobs WHERE archived=0 AND status IN ('queued','sending')").fetchone()[0],
            "sent": db.execute("SELECT COUNT(*) FROM return_jobs WHERE archived=0 AND status='sent'").fetchone()[0],
            "failed": db.execute("SELECT COUNT(*) FROM return_jobs WHERE archived=0 AND status='failed'").fetchone()[0],
        }


def audit(action: str, object_type: str, object_id: object, detail: str = "", actor: str = "system") -> None:
    with connect() as db:
        db.execute("INSERT INTO audit_events(actor,action,object_type,object_id,detail) VALUES(?,?,?,?,?)",
                   (actor, action, object_type, str(object_id), detail[:2000]))


def setting(key: str, default: str = "0") -> str:
    with connect() as db:
        row = db.execute("SELECT value FROM system_settings WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default


def set_setting(key: str, value: str) -> None:
    with connect() as db:
        db.execute("INSERT INTO system_settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP", (key, value))


def authorize(ip: str, aet: str, called_aet: str = "") -> bool:
    with connect() as db:
        allowed = db.execute(
            "SELECT 1 FROM centers WHERE source_ip=? AND source_aet=? AND enabled=1",
            (ip, aet),
        ).fetchone() is not None
    if not allowed:
        record_pending_sender(ip, aet, called_aet)
    return allowed
