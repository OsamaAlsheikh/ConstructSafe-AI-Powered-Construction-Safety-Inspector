# # database.py  [ROOT FOLDER]
# # SQLite persistence layer. Auto-creates schema on first run.
# # Tables: inspections (one row per inspection run)
# #         violations  (one row per violation found in each inspection)

# import sqlite3, logging
# from datetime import datetime, timedelta
# from pathlib import Path
# from config import DB_PATH, REPEAT_VIOLATION_WINDOW_DAYS

# logger = logging.getLogger(__name__)

# def _get_connection() -> sqlite3.Connection:
#     conn = sqlite3.connect(str(DB_PATH))
#     conn.row_factory = sqlite3.Row   # row['column'] access
#     _create_tables(conn)
#     return conn

# def _create_tables(conn):
#     conn.executescript('''
#         CREATE TABLE IF NOT EXISTS inspections (
#             id              INTEGER PRIMARY KEY AUTOINCREMENT,
#             site_name       TEXT NOT NULL,
#             inspection_date TEXT NOT NULL,
#             risk_level      TEXT NOT NULL,
#             violation_count INTEGER NOT NULL,
#             workers_observed INTEGER,
#             report_text     TEXT,
#             supervisor_email TEXT,
#             created_at      TEXT NOT NULL
#         );
#         CREATE TABLE IF NOT EXISTS violations (
#             id              INTEGER PRIMARY KEY AUTOINCREMENT,
#             inspection_id   INTEGER NOT NULL,
#             violation_code  TEXT NOT NULL,
#             violation_name  TEXT,
#             regulation      TEXT,
#             severity        TEXT,
#             observation     TEXT,
#             location_description TEXT,
#             confidence      REAL,
#             FOREIGN KEY (inspection_id) REFERENCES inspections(id)
#         );
#         CREATE INDEX IF NOT EXISTS idx_site ON inspections(site_name);
#         CREATE INDEX IF NOT EXISTS idx_date ON inspections(inspection_date);
#         CREATE INDEX IF NOT EXISTS idx_code ON violations(violation_code);
#     ''')
#     conn.commit()

# def log_inspection(site_name, violations, risk_level,
#                    report_text, supervisor_email, workers_observed=0) -> int:
#     """Insert one inspection + its violations. Returns new inspection ID."""
#     conn = _get_connection()
#     now  = datetime.now().isoformat()
#     try:
#         cur = conn.execute(
#             'INSERT INTO inspections (site_name,inspection_date,risk_level,violation_count,'
#             'workers_observed,report_text,supervisor_email,created_at) VALUES (?,?,?,?,?,?,?,?)',
#             (site_name,now,risk_level,len(violations),workers_observed,report_text,supervisor_email,now))
#         iid = cur.lastrowid
#         for v in violations:
#             conn.execute(
#                 'INSERT INTO violations (inspection_id,violation_code,violation_name,regulation,'
#                 'severity,observation,location_description,confidence) VALUES (?,?,?,?,?,?,?,?)',
#                 (iid,v.get('code','?'),v.get('name',''),v.get('regulation',''),
#                  v.get('severity',''),v.get('observation',''),
#                  v.get('location_description',''),v.get('confidence',0.0)))
#         conn.commit()
#         return iid
#     finally: conn.close()

# def get_repeat_violation_count(site_name: str) -> int:
#     """Count inspections-with-violations in the last REPEAT_VIOLATION_WINDOW_DAYS days."""
#     conn   = _get_connection()
#     cutoff = (datetime.now() - timedelta(days=REPEAT_VIOLATION_WINDOW_DAYS)).isoformat()
#     try:
#         row = conn.execute(
#             'SELECT COUNT(*) as cnt FROM inspections WHERE site_name=? AND inspection_date>=? AND violation_count>0',
#             (site_name, cutoff)).fetchone()
#         return row['cnt'] if row else 0
#     finally: conn.close()

# def get_site_history(site_name: str, limit: int=10) -> list:
#     conn = _get_connection()
#     try:
#         rows = conn.execute(
#             'SELECT inspection_date,risk_level,violation_count,workers_observed '
#             'FROM inspections WHERE site_name=? ORDER BY inspection_date DESC LIMIT ?',
#             (site_name, limit)).fetchall()
#         return [dict(r) for r in rows]
#     finally: conn.close()

# def get_dashboard_stats() -> dict:
#     conn = _get_connection()
#     try:
#         total = conn.execute('SELECT COUNT(*) as c FROM inspections').fetchone()['c']
#         viols = conn.execute('SELECT SUM(violation_count) as s FROM inspections').fetchone()['s'] or 0
#         high  = conn.execute("SELECT COUNT(*) as c FROM inspections WHERE risk_level IN ('HIGH','CRITICAL')").fetchone()['c']
#         sites = conn.execute(
#             'SELECT site_name,COUNT(*) as insp,SUM(violation_count) as tv,MAX(inspection_date) as li '
#             'FROM inspections GROUP BY site_name ORDER BY li DESC LIMIT 5').fetchall()
#         return {'total_inspections':total,'total_violations':int(viols),
#                 'high_risk_inspections':high,'recent_sites':[dict(r) for r in sites]}
#     finally: conn.close()



# database.py
import sqlite3
import logging
from datetime import datetime, timedelta
from config import DB_PATH, REPEAT_VIOLATION_WINDOW_DAYS

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS inspections (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            site_name         TEXT NOT NULL,
            inspection_date   TEXT NOT NULL,
            risk_level        TEXT NOT NULL,
            violation_count   INTEGER NOT NULL,
            workers_observed  INTEGER,
            report_text       TEXT,
            supervisor_email  TEXT,
            created_at        TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS violations (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id        INTEGER NOT NULL,
            violation_code       TEXT NOT NULL,
            violation_name       TEXT,
            regulation           TEXT,
            severity             TEXT,
            observation          TEXT,
            location_description TEXT,
            confidence           REAL,
            FOREIGN KEY (inspection_id) REFERENCES inspections(id)
        );
        CREATE INDEX IF NOT EXISTS idx_site ON inspections(site_name);
        CREATE INDEX IF NOT EXISTS idx_date ON inspections(inspection_date);
        CREATE INDEX IF NOT EXISTS idx_code ON violations(violation_code);
    ''')
    conn.commit()


def log_inspection(
    site_name: str,
    violations: list,
    risk_level: str,
    report_text: str,
    supervisor_email: str,
    workers_observed: int = 0,
) -> int:
    conn = _get_connection()
    now  = datetime.now().isoformat()
    try:
        cur = conn.execute(
            'INSERT INTO inspections (site_name, inspection_date, risk_level,'
            ' violation_count, workers_observed, report_text, supervisor_email, created_at)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (site_name, now, risk_level, len(violations),
             workers_observed, report_text, supervisor_email, now),
        )
        inspection_id = cur.lastrowid
        for v in violations:
            conn.execute(
                'INSERT INTO violations (inspection_id, violation_code, violation_name,'
                ' regulation, severity, observation, location_description, confidence)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (inspection_id, v.get('code', '?'), v.get('name', ''),
                 v.get('regulation', ''), v.get('severity', ''),
                 v.get('observation', ''), v.get('location_description', ''),
                 v.get('confidence', 0.0)),
            )
        conn.commit()
        return inspection_id
    finally:
        conn.close()


def get_repeat_violation_count(site_name: str) -> int:
    conn   = _get_connection()
    cutoff = (datetime.now() - timedelta(days=REPEAT_VIOLATION_WINDOW_DAYS)).isoformat()
    try:
        row = conn.execute(
            'SELECT COUNT(*) as cnt FROM inspections'
            ' WHERE site_name=? AND inspection_date>=? AND violation_count>0',
            (site_name, cutoff),
        ).fetchone()
        return row['cnt'] if row else 0
    finally:
        conn.close()


def get_site_history(site_name: str, limit: int = 10) -> list:
    conn = _get_connection()
    try:
        rows = conn.execute(
            'SELECT inspection_date, risk_level, violation_count, workers_observed'
            ' FROM inspections WHERE site_name=?'
            ' ORDER BY inspection_date DESC LIMIT ?',
            (site_name, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_dashboard_stats() -> dict:
    conn = _get_connection()
    try:
        total = conn.execute(
            'SELECT COUNT(*) as c FROM inspections'
        ).fetchone()['c']
        viols = conn.execute(
            'SELECT SUM(violation_count) as s FROM inspections'
        ).fetchone()['s'] or 0
        high  = conn.execute(
            "SELECT COUNT(*) as c FROM inspections WHERE risk_level IN ('HIGH','CRITICAL')"
        ).fetchone()['c']
        sites = conn.execute(
            'SELECT site_name, COUNT(*) as insp, SUM(violation_count) as tv,'
            ' MAX(inspection_date) as li FROM inspections'
            ' GROUP BY site_name ORDER BY li DESC LIMIT 5'
        ).fetchall()
        return {
            'total_inspections':     total,
            'total_violations':      int(viols),
            'high_risk_inspections': high,
            'recent_sites':          [dict(r) for r in sites],
        }
    finally:
        conn.close()