"""SQLite persistence for ABSTRAL runs.

Stores run metadata, iteration results, and case-level data
for comparison across experiments.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from runner.agent_system import IterResult, CaseResult


def _default_db_path() -> str:
    data_dir = os.environ.get("DATA_DIR", "data")
    return f"{data_dir}/abstral_runs.db"


class RunDB:
    """Lightweight SQLite database for persisting ABSTRAL runs."""

    def __init__(self, path: str = None):
        if path is None:
            path = _default_db_path()
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at REAL,
                config TEXT,
                status TEXT DEFAULT 'running',
                best_auc REAL DEFAULT 0,
                best_iter INTEGER DEFAULT -1,
                total_iterations INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                topology TEXT,
                agents TEXT,
                rationale TEXT,
                metrics TEXT,
                spec TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                patient_id TEXT,
                risk_score REAL,
                label INTEGER,
                ground_truth INTEGER,
                correct INTEGER,
                tokens INTEGER,
                topology TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_iterations_run ON iterations(run_id);
            CREATE INDEX IF NOT EXISTS idx_cases_run_iter ON cases(run_id, iteration);

            CREATE TABLE IF NOT EXISTS baseline_runs (
                id TEXT PRIMARY KEY,
                created_at REAL,
                config TEXT,
                status TEXT DEFAULT 'running',
                results TEXT
            );

            CREATE TABLE IF NOT EXISTS matrix_runs (
                id TEXT PRIMARY KEY,
                created_at REAL,
                config TEXT,
                status TEXT DEFAULT 'running',
                results TEXT
            );
        """)
        self.conn.commit()

    def create_run(self, config: dict, name: str = None) -> str:
        """Create a new run. Returns run_id."""
        run_id = str(uuid.uuid4())[:8]
        if not name:
            name = f"run_{run_id}"
        self.conn.execute(
            "INSERT INTO runs (id, name, created_at, config, status) VALUES (?, ?, ?, ?, ?)",
            (run_id, name, time.time(), json.dumps(config, default=str), "running")
        )
        self.conn.commit()
        return run_id

    def update_run(self, run_id: str, **kwargs):
        """Update run fields (status, best_auc, best_iter, total_iterations)."""
        valid_fields = {"status", "best_auc", "best_iter", "total_iterations", "name"}
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [run_id]
        self.conn.execute(f"UPDATE runs SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    def save_iteration(self, run_id: str, iter_result: IterResult):
        """Save an iteration's results."""
        spec_dict = iter_result.spec.to_dict() if iter_result.spec else {}
        self.conn.execute(
            "INSERT INTO iterations (run_id, iteration, topology, agents, rationale, metrics, spec) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                iter_result.iteration,
                iter_result.spec.topology if iter_result.spec else "",
                json.dumps(iter_result.spec.agent_ids if iter_result.spec else []),
                iter_result.spec.rationale if iter_result.spec else "",
                json.dumps(iter_result.metrics, default=str),
                json.dumps(spec_dict, default=str),
            )
        )
        self.conn.commit()

    def save_cases(self, run_id: str, iteration: int, case_results: list[CaseResult]):
        """Bulk save case results for an iteration."""
        rows = []
        for cr in case_results:
            rows.append((
                run_id, iteration, cr.patient_id,
                cr.prediction.get("risk_score", 0),
                cr.prediction.get("label", 0),
                cr.prediction.get("ground_truth", -1),
                int(cr.correct),
                cr.total_tokens,
                cr.topology,
            ))
        self.conn.executemany(
            "INSERT INTO cases (run_id, iteration, patient_id, risk_score, "
            "label, ground_truth, correct, tokens, topology) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get run details."""
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config"] = json.loads(d["config"]) if d["config"] else {}
        return d

    def list_runs(self) -> list[dict]:
        """List all runs, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["config"] = json.loads(d["config"]) if d["config"] else {}
            results.append(d)
        return results

    def get_iterations(self, run_id: str) -> list[dict]:
        """Get all iterations for a run."""
        rows = self.conn.execute(
            "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration",
            (run_id,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["agents"] = json.loads(d["agents"]) if d["agents"] else []
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            d["spec"] = json.loads(d["spec"]) if d["spec"] else {}
            results.append(d)
        return results

    def get_best_spec(self, run_id: str) -> Optional[dict]:
        """Get the AgentSpec from the best-performing iteration."""
        run = self.get_run(run_id)
        if not run or run["best_iter"] < 0:
            return None
        row = self.conn.execute(
            "SELECT spec, metrics FROM iterations WHERE run_id = ? AND iteration = ?",
            (run_id, run["best_iter"])
        ).fetchone()
        if not row:
            return None
        spec = json.loads(row["spec"]) if row["spec"] else {}
        spec["_metrics"] = json.loads(row["metrics"]) if row["metrics"] else {}
        return spec

    def get_cases(self, run_id: str, iteration: int) -> list[dict]:
        """Get case results for a specific iteration."""
        rows = self.conn.execute(
            "SELECT * FROM cases WHERE run_id = ? AND iteration = ? ORDER BY patient_id",
            (run_id, iteration)
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_run(self, run_id: str):
        """Delete a run and all its data."""
        self.conn.execute("DELETE FROM cases WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM iterations WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self.conn.commit()

    # ── Baseline Runs ────────────────────────────────────────────────────────

    def save_baseline_run(self, config: dict, results: dict) -> str:
        """Save a baseline comparison run. Returns run_id."""
        run_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            "INSERT INTO baseline_runs (id, created_at, config, status, results) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, time.time(), json.dumps(config, default=str),
             "completed", json.dumps(results, default=str))
        )
        self.conn.commit()
        return run_id

    def update_baseline_run(self, run_id: str, **kwargs):
        """Update baseline run fields (status, results)."""
        valid = {"status", "results"}
        updates = {k: (json.dumps(v, default=str) if k == "results" else v)
                   for k, v in kwargs.items() if k in valid}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [run_id]
        self.conn.execute(f"UPDATE baseline_runs SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    def list_baseline_runs(self) -> list[dict]:
        """List all baseline runs, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM baseline_runs ORDER BY created_at DESC"
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["config"] = json.loads(d["config"]) if d["config"] else {}
            d["results"] = json.loads(d["results"]) if d["results"] else {}
            out.append(d)
        return out

    def get_baseline_run(self, run_id: str) -> Optional[dict]:
        """Get a specific baseline run."""
        row = self.conn.execute(
            "SELECT * FROM baseline_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config"] = json.loads(d["config"]) if d["config"] else {}
        d["results"] = json.loads(d["results"]) if d["results"] else {}
        return d

    def delete_baseline_run(self, run_id: str):
        self.conn.execute("DELETE FROM baseline_runs WHERE id = ?", (run_id,))
        self.conn.commit()

    # ── Matrix Runs ──────────────────────────────────────────────────────────

    def save_matrix_run(self, config: dict, results: dict) -> str:
        """Save a topology matrix run. Returns run_id."""
        run_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            "INSERT INTO matrix_runs (id, created_at, config, status, results) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, time.time(), json.dumps(config, default=str),
             "completed", json.dumps(results, default=str))
        )
        self.conn.commit()
        return run_id

    def update_matrix_run(self, run_id: str, **kwargs):
        """Update matrix run fields (status, results)."""
        valid = {"status", "results"}
        updates = {k: (json.dumps(v, default=str) if k == "results" else v)
                   for k, v in kwargs.items() if k in valid}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [run_id]
        self.conn.execute(f"UPDATE matrix_runs SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    def list_matrix_runs(self) -> list[dict]:
        """List all matrix runs, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM matrix_runs ORDER BY created_at DESC"
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["config"] = json.loads(d["config"]) if d["config"] else {}
            d["results"] = json.loads(d["results"]) if d["results"] else {}
            out.append(d)
        return out

    def get_matrix_run(self, run_id: str) -> Optional[dict]:
        """Get a specific matrix run."""
        row = self.conn.execute(
            "SELECT * FROM matrix_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config"] = json.loads(d["config"]) if d["config"] else {}
        d["results"] = json.loads(d["results"]) if d["results"] else {}
        return d

    def delete_matrix_run(self, run_id: str):
        self.conn.execute("DELETE FROM matrix_runs WHERE id = ?", (run_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()
