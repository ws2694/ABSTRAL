#!/usr/bin/env python3
"""ABSTRAL Web Dashboard — FastAPI server for collaborative interaction.

Usage:
    python server.py                  # start on port 8420
    python server.py --port 3000      # custom port
    uvicorn server:app --reload       # dev mode with hot reload
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import difflib as _difflib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# ── Event Bus ─────────────────────────────────────────────────────────────────

class EventBus:
    """Pub-sub event bus for streaming run progress to WebSocket clients."""

    def __init__(self):
        self.events: list[dict] = []
        self._subscribers: list = []

    def emit(self, event_type: str, data: Any = None):
        event = {
            "type": event_type,
            "data": data or {},
            "ts": time.time()
        }
        self.events.append(event)
        for cb in self._subscribers:
            try:
                asyncio.ensure_future(cb(event))
            except RuntimeError:
                pass

    def subscribe(self, callback):
        self._subscribers.append(callback)

    def unsubscribe(self, callback):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def clear(self):
        self.events.clear()


# ── Run State ─────────────────────────────────────────────────────────────────

class RunState:
    """Mutable shared state for the current run."""

    def __init__(self):
        self.status: str = "idle"  # idle | running | completed | stopped | error
        self.config: dict = {}
        self.iterations: list[dict] = []
        self.logs: list[dict] = []
        self.current_iteration: int = -1
        self.total_iterations: int = 15
        self.best_auc: float = 0.0
        self.best_iter: int = -1
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        # Per-iteration detail storage
        self.analyses: dict[int, dict] = {}       # iter → analysis findings
        self.skill_updates: dict[int, list] = {}  # iter → diff_log
        self.agent_specs: dict[int, dict] = {}    # iter → compiled spec
        self.step_status: dict[int, dict] = {}    # iter → {build, run, eval, analyze, update}

    def add_log(self, level: str, message: str, event_bus: EventBus = None):
        entry = {"level": level, "message": message, "ts": time.time()}
        self.logs.append(entry)
        if event_bus:
            event_bus.emit("log", entry)

    def reset(self):
        self.status = "idle"
        self.iterations = []
        self.logs = []
        self.current_iteration = -1
        self.best_auc = 0.0
        self.best_iter = -1
        self.error = None
        self._task = None
        self.analyses = {}
        self.skill_updates = {}
        self.agent_specs = {}
        self.step_status = {}


event_bus = EventBus()
run_state = RunState()

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="ABSTRAL Dashboard", version="1.0.0")

_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth Middleware ───────────────────────────────────────────────────────────

_AUTH_EXEMPT = {"/", "/api/status", "/api/auth/check", "/api/auth/login"}


@app.middleware("http")
async def auth_middleware(request, call_next):
    """Simple shared-password auth. Disabled if ABSTRAL_AUTH_TOKEN is unset."""
    token = os.environ.get("ABSTRAL_AUTH_TOKEN")
    if not token:
        return await call_next(request)
    # Exempt paths (health check, landing page, auth check)
    if request.url.path in _AUTH_EXEMPT:
        return await call_next(request)
    # Check Authorization header or ?token= query param
    auth = request.headers.get("authorization", "")
    qtoken = request.query_params.get("token", "")
    if auth == f"Bearer {token}" or qtoken == token:
        return await call_next(request)
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


@app.get("/api/auth/check")
async def auth_check():
    """Check if auth is enabled and validate a token."""
    token = os.environ.get("ABSTRAL_AUTH_TOKEN")
    if not token:
        return {"auth_required": False, "valid": True}
    return {"auth_required": True}


@app.post("/api/auth/login")
async def auth_login(body: dict):
    """Validate password and return success."""
    token = os.environ.get("ABSTRAL_AUTH_TOKEN")
    if not token:
        return {"valid": True}
    if body.get("password") == token:
        return {"valid": True, "token": token}
    return JSONResponse({"error": "Invalid password"}, status_code=401)


# ── Page Routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("webapp/index.html", media_type="text/html")


# ── API: Status ───────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "status": run_state.status,
        "current_iteration": run_state.current_iteration,
        "total_iterations": run_state.total_iterations,
        "best_auc": run_state.best_auc,
        "best_iter": run_state.best_iter,
        "iterations_completed": len(run_state.iterations),
        "error": run_state.error,
    }


# ── API: Config ───────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    if run_state.config:
        return run_state.config
    _dd = os.environ.get("DATA_DIR", "data")
    return {
        "data_path": f"{_dd}/oncoagent_7315.parquet",
        "skill_path": "skills/clinical_agent_builder.md",
        "max_iterations": 15,
        "sandbox_n": 150,
        "model": "claude-sonnet-4-20250514",
        "agent_model": "claude-sonnet-4-20250514",
        "model_dir": f"{_dd}/models",
        "random_seed": 42,
        "max_concurrent": 10,
        "use_batch_api": False,
    }


@app.post("/api/config")
async def update_config(body: dict):
    if run_state.status == "running":
        return JSONResponse({"error": "Cannot update config while running"}, 400)
    run_state.config.update(body)
    return {"ok": True}


# ── API: Run Control ──────────────────────────────────────────────────────────

@app.post("/api/run/start")
async def start_run(body: dict = None):
    if run_state.status == "running":
        return JSONResponse({"error": "Already running"}, 400)

    config_dict = {**(await get_config()), **(body or {})}
    run_state.reset()
    run_state.config = config_dict
    run_state.status = "running"
    run_state.total_iterations = config_dict.get("max_iterations", 15)
    event_bus.clear()

    run_state._task = asyncio.create_task(_run_abstral_task(config_dict))

    event_bus.emit("run_started", config_dict)
    run_state.add_log("info", "Run started", event_bus)
    return {"ok": True}


@app.post("/api/run/stop")
async def stop_run():
    if run_state.status != "running":
        return JSONResponse({"error": "Not running"}, 400)

    run_state.status = "stopped"
    if run_state._task and not run_state._task.done():
        run_state._task.cancel()
    event_bus.emit("run_stopped", {})
    run_state.add_log("warn", "Run stopped by user", event_bus)
    return {"ok": True}


# ── API: Iterations ───────────────────────────────────────────────────────────

@app.get("/api/iterations")
async def get_iterations():
    return run_state.iterations


@app.get("/api/iterations/{iteration}")
async def get_iteration(iteration: int):
    for it in run_state.iterations:
        if it.get("iteration") == iteration:
            return it
    return JSONResponse({"error": "Iteration not found"}, 404)


# ── API: Iteration Detail ─────────────────────────────────────────────────────

@app.get("/api/iterations/{iteration}/analysis")
async def get_iteration_analysis(iteration: int):
    return run_state.analyses.get(iteration, {"findings": [], "evidence_classes": []})


@app.get("/api/iterations/{iteration}/updates")
async def get_iteration_updates(iteration: int):
    return run_state.skill_updates.get(iteration, [])


@app.get("/api/iterations/{iteration}/spec")
async def get_iteration_spec(iteration: int):
    return run_state.agent_specs.get(iteration, {})


@app.get("/api/iterations/{iteration}/steps")
async def get_iteration_steps(iteration: int):
    return run_state.step_status.get(iteration, {})


# ── API: Traces ───────────────────────────────────────────────────────────────

@app.get("/api/traces/{iteration}")
async def get_traces(iteration: int):
    trace_dir = Path("traces") / f"iter_{iteration:03d}"
    if not trace_dir.exists():
        return []
    traces = []
    for f in sorted(trace_dir.glob("P*.json")):
        try:
            traces.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, IOError):
            pass
    if not traces:
        # Fallback to case_*.json pattern
        for f in sorted(trace_dir.glob("case_*.json")):
            try:
                traces.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, IOError):
                pass
    return traces


@app.get("/api/traces/{iteration}/{patient_id}")
async def get_trace(iteration: int, patient_id: str):
    trace_dir = Path("traces") / f"iter_{iteration:03d}"
    for f in trace_dir.glob("case_*.json"):
        try:
            trace = json.loads(f.read_text())
            if trace.get("meta", {}).get("patient_id") == patient_id:
                return trace
        except (json.JSONDecodeError, IOError):
            pass
    return JSONResponse({"error": "Trace not found"}, 404)


# ── API: Skill ────────────────────────────────────────────────────────────────

@app.get("/api/skill")
async def get_skill():
    skill_path = run_state.config.get("skill_path", "skills/clinical_agent_builder.md")
    p = Path(skill_path)
    if p.exists():
        return {"content": p.read_text(), "path": str(p)}
    return JSONResponse({"error": "Skill file not found"}, 404)


@app.put("/api/skill")
async def update_skill(body: dict):
    if run_state.status == "running":
        return JSONResponse({"error": "Cannot edit skill while running"}, 400)
    skill_path = run_state.config.get("skill_path", "skills/clinical_agent_builder.md")
    Path(skill_path).write_text(body["content"])
    run_state.add_log("info", "Skill updated manually", event_bus)
    return {"ok": True}


@app.get("/api/skill/versions")
async def get_skill_versions():
    versions_dir = Path("skills/versions")
    if not versions_dir.exists():
        return []
    versions = sorted(versions_dir.glob("ABS_v*.md"))
    return [{"name": v.name, "path": str(v), "size": v.stat().st_size} for v in versions]


@app.get("/api/skill/versions/{name}")
async def get_skill_version(name: str):
    p = Path("skills/versions") / name
    if p.exists():
        return {"content": p.read_text(), "name": name}
    return JSONResponse({"error": "Version not found"}, 404)


@app.get("/api/skill/diff/{v1}/{v2}")
async def get_skill_diff(v1: str, v2: str):
    """Get a unified diff between two skill versions."""
    import difflib
    p1 = Path("skills/versions") / v1
    p2 = Path("skills/versions") / v2
    if not p1.exists() or not p2.exists():
        return JSONResponse({"error": "Version not found"}, 404)
    lines1 = p1.read_text().splitlines(keepends=True)
    lines2 = p2.read_text().splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2, fromfile=v1, tofile=v2, n=3))
    return {"diff": "".join(diff), "v1": v1, "v2": v2}


# ── API: Logs ─────────────────────────────────────────────────────────────────

@app.get("/api/logs")
async def get_logs():
    return run_state.logs[-500:]


# ── API: Prerequisites ────────────────────────────────────────────────────────

@app.get("/api/prerequisites")
async def check_prerequisites():
    import os
    checks = []
    config = await get_config()

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    checks.append({"name": "API Key", "ok": has_key,
                    "detail": "Set" if has_key else "ANTHROPIC_API_KEY not set"})

    data_ok = Path(config["data_path"]).exists()
    checks.append({"name": "Patient Data", "ok": data_ok,
                    "detail": config["data_path"]})

    model_dir = Path(config["model_dir"])
    models_ok = model_dir.exists() and any(model_dir.glob("*.pkl"))
    checks.append({"name": "Trained Models", "ok": models_ok,
                    "detail": config["model_dir"]})

    skill_ok = Path(config["skill_path"]).exists()
    checks.append({"name": "Skill File", "ok": skill_ok,
                    "detail": config["skill_path"]})

    return {"checks": checks, "all_ok": all(c["ok"] for c in checks)}


# ── API: Runs (Database) ─────────────────────────────────────────────────────

def _get_db():
    from db import RunDB
    return RunDB()


@app.get("/api/runs")
async def list_runs():
    db = _get_db()
    runs = db.list_runs()
    db.close()
    return runs


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    db = _get_db()
    run = db.get_run(run_id)
    db.close()
    if not run:
        return JSONResponse({"error": "Run not found"}, 404)
    return run


@app.get("/api/runs/{run_id}/iterations")
async def get_run_iterations(run_id: str):
    db = _get_db()
    iterations = db.get_iterations(run_id)
    db.close()
    return iterations


@app.get("/api/runs/{run_id}/best-spec")
async def get_run_best_spec(run_id: str):
    db = _get_db()
    spec = db.get_best_spec(run_id)
    db.close()
    if not spec:
        return JSONResponse({"error": "No best spec found"}, 404)
    return spec


@app.get("/api/runs/{run_id}/best-spec/download")
async def download_best_spec(run_id: str):
    db = _get_db()
    spec = db.get_best_spec(run_id)
    db.close()
    if not spec:
        return JSONResponse({"error": "No best spec found"}, 404)
    content = json.dumps(spec, indent=2, default=str)
    return JSONResponse(
        content=json.loads(content),
        headers={"Content-Disposition": f"attachment; filename=best_spec_{run_id}.json"}
    )


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    db = _get_db()
    db.delete_run(run_id)
    db.close()
    return {"ok": True}


@app.get("/api/runs/{run_id}/summary")
async def get_run_summary(run_id: str):
    db = _get_db()
    run = db.get_run(run_id)
    if not run:
        db.close()
        return JSONResponse({"error": "Run not found"}, 404)
    iterations = db.get_iterations(run_id)
    best_spec = db.get_best_spec(run_id)
    db.close()

    from eval.report import generate_run_summary
    return generate_run_summary(run, iterations, best_spec)


# ── API: Topology Export ─────────────────────────────────────────────────────

@app.get("/api/iterations/{iteration}/spec/download")
async def download_iteration_spec(iteration: int):
    spec = run_state.agent_specs.get(iteration)
    if not spec:
        return JSONResponse({"error": "Spec not found"}, 404)
    content = json.dumps(spec, indent=2, default=str)
    return JSONResponse(
        content=json.loads(content),
        headers={"Content-Disposition": f"attachment; filename=spec_iter_{iteration}.json"}
    )


# ── API: Baselines ──────────────────────────────────────────────────────────

# Track running baseline/matrix tasks
_baseline_task: Optional[asyncio.Task] = None
_matrix_task: Optional[asyncio.Task] = None
_baseline_status = {"status": "idle", "run_id": None}
_matrix_status = {"status": "idle", "run_id": None}


@app.post("/api/baselines/run")
async def run_baselines_endpoint(body: dict = None):
    global _baseline_task, _baseline_status
    if _baseline_status["status"] == "running":
        return JSONResponse({"error": "Baselines already running"}, 400)

    body = body or {}
    _baseline_status = {"status": "running", "run_id": None}
    _baseline_task = asyncio.create_task(_run_baselines_task(body))
    return {"ok": True, "message": "Baselines started"}


async def _run_baselines_task(body: dict):
    global _baseline_status
    try:
        from runner.sandbox import PatientStore
        from baselines.runner import run_all_baselines

        data_path = body.get("data_path", "data/oncoagent.parquet")
        model_dir = body.get("model_dir", "data/models")
        model = body.get("model", "claude-sonnet-4-20250514")
        sandbox_n = body.get("sandbox_n", 150)
        seed = body.get("seed", 42)
        baselines_list = body.get("baselines")  # None = all

        config = {
            "data_path": data_path, "model_dir": model_dir, "model": model,
            "sandbox_n": sandbox_n, "seed": seed, "baselines": baselines_list,
        }

        # Create DB record early
        db = _get_db()
        run_id = db.save_baseline_run(config, {})
        db.update_baseline_run(run_id, status="running")
        db.close()
        _baseline_status["run_id"] = run_id

        event_bus.emit("baseline_started", {"run_id": run_id, "config": config})

        patient_store = PatientStore.load(data_path, model_dir)
        patient_ids = patient_store.stratified_sample(n=sandbox_n, seed=seed)

        results = await run_all_baselines(
            patient_ids=patient_ids,
            patient_store=patient_store,
            model=model,
            baselines=baselines_list,
        )

        # Strip case_results for storage
        stored = {}
        for name, data in results.items():
            stored[name] = {
                "metrics": data["metrics"],
                "total_tokens": data["total_tokens"],
                "n_cases": len(data.get("case_results", [])),
            }

        db = _get_db()
        db.update_baseline_run(run_id, status="completed", results=stored)
        db.close()

        _baseline_status = {"status": "completed", "run_id": run_id}
        event_bus.emit("baseline_complete", {"run_id": run_id, "results": stored})

    except Exception as e:
        if _baseline_status.get("run_id"):
            db = _get_db()
            db.update_baseline_run(_baseline_status["run_id"], status="error")
            db.close()
        _baseline_status = {"status": "error", "run_id": _baseline_status.get("run_id"),
                            "error": str(e)}
        event_bus.emit("baseline_error", {"error": str(e)})


@app.get("/api/baselines/status")
async def get_baselines_status():
    return _baseline_status


@app.get("/api/baselines/results")
async def list_baseline_results():
    db = _get_db()
    runs = db.list_baseline_runs()
    db.close()
    return runs


@app.get("/api/baselines/results/{run_id}")
async def get_baseline_result(run_id: str):
    db = _get_db()
    run = db.get_baseline_run(run_id)
    db.close()
    if not run:
        return JSONResponse({"error": "Baseline run not found"}, 404)
    return run


@app.delete("/api/baselines/results/{run_id}")
async def delete_baseline_result(run_id: str):
    db = _get_db()
    db.delete_baseline_run(run_id)
    db.close()
    return {"ok": True}


# ── API: Topology Matrix ─────────────────────────────────────────────────────

@app.post("/api/matrix/run")
async def run_matrix_endpoint(body: dict = None):
    global _matrix_task, _matrix_status
    if _matrix_status["status"] == "running":
        return JSONResponse({"error": "Matrix already running"}, 400)

    body = body or {}
    _matrix_status = {"status": "running", "run_id": None}
    _matrix_task = asyncio.create_task(_run_matrix_task(body))
    return {"ok": True, "message": "Matrix started"}


async def _run_matrix_task(body: dict):
    global _matrix_status
    try:
        from runner.sandbox import PatientStore
        from runner.agent_system import AgentSpec
        from baselines.topology_matrix import run_topology_matrix

        data_path = body.get("data_path", "data/oncoagent.parquet")
        model_dir = body.get("model_dir", "data/models")
        sandbox_n = body.get("sandbox_n", 50)
        seed = body.get("seed", 42)
        models = body.get("models", ["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"])

        # Get spec: either from body or from a run_id
        spec_dict = body.get("spec")
        source_run_id = body.get("source_run_id")
        if not spec_dict and source_run_id:
            db = _get_db()
            spec_dict = db.get_best_spec(source_run_id)
            db.close()
        if not spec_dict:
            _matrix_status = {"status": "error", "error": "No spec provided"}
            return

        spec_dict.pop("_metrics", None)
        spec = AgentSpec.from_dict(spec_dict)

        config = {
            "data_path": data_path, "model_dir": model_dir,
            "sandbox_n": sandbox_n, "seed": seed,
            "models": models, "topology": spec.topology,
            "source_run_id": source_run_id,
        }

        db = _get_db()
        run_id = db.save_matrix_run(config, {})
        db.update_matrix_run(run_id, status="running")
        db.close()
        _matrix_status["run_id"] = run_id

        event_bus.emit("matrix_started", {"run_id": run_id, "config": config})

        patient_store = PatientStore.load(data_path, model_dir)
        patient_ids = patient_store.stratified_sample(n=sandbox_n, seed=seed)

        results = await run_topology_matrix(
            spec=spec,
            patient_ids=patient_ids,
            patient_store=patient_store,
            models=models,
        )

        db = _get_db()
        db.update_matrix_run(run_id, status="completed", results=results)
        db.close()

        _matrix_status = {"status": "completed", "run_id": run_id}
        event_bus.emit("matrix_complete", {"run_id": run_id, "results": results})

    except Exception as e:
        if _matrix_status.get("run_id"):
            db = _get_db()
            db.update_matrix_run(_matrix_status["run_id"], status="error")
            db.close()
        _matrix_status = {"status": "error", "run_id": _matrix_status.get("run_id"),
                          "error": str(e)}
        event_bus.emit("matrix_error", {"error": str(e)})


@app.get("/api/matrix/status")
async def get_matrix_status():
    return _matrix_status


@app.get("/api/matrix/results")
async def list_matrix_results():
    db = _get_db()
    runs = db.list_matrix_runs()
    db.close()
    return runs


@app.get("/api/matrix/results/{run_id}")
async def get_matrix_result(run_id: str):
    db = _get_db()
    run = db.get_matrix_run(run_id)
    db.close()
    if not run:
        return JSONResponse({"error": "Matrix run not found"}, 404)
    return run


@app.delete("/api/matrix/results/{run_id}")
async def delete_matrix_result(run_id: str):
    db = _get_db()
    db.delete_matrix_run(run_id)
    db.close()
    return {"ok": True}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Auth check for WebSocket (token via query param)
    token = os.environ.get("ABSTRAL_AUTH_TOKEN")
    if token:
        qtoken = websocket.query_params.get("token", "")
        if qtoken != token:
            await websocket.close(code=4001, reason="Unauthorized")
            return
    await websocket.accept()

    # Send current state on connect
    await websocket.send_json({
        "type": "state_sync",
        "data": {
            "status": run_state.status,
            "iterations": run_state.iterations,
            "logs": run_state.logs[-200:],
            "best_auc": run_state.best_auc,
            "best_iter": run_state.best_iter,
            "current_iteration": run_state.current_iteration,
            "total_iterations": run_state.total_iterations,
            "analyses": run_state.analyses,
            "skill_updates": run_state.skill_updates,
            "agent_specs": run_state.agent_specs,
            "step_status": run_state.step_status,
        }
    })

    async def forward_event(event):
        try:
            await websocket.send_json(event)
        except Exception:
            pass

    event_bus.subscribe(forward_event)

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        event_bus.unsubscribe(forward_event)
    except Exception:
        event_bus.unsubscribe(forward_event)


# ── Background Run Task ──────────────────────────────────────────────────────

async def _run_abstral_task(config_dict: dict):
    """Run the ABSTRAL meta-loop as a background async task."""
    try:
        from config import ABSTRALConfig
        from loop.orchestrator import run_abstral

        config = ABSTRALConfig(
            data_path=config_dict.get("data_path", "data/oncoagent.parquet"),
            skill_path=config_dict.get("skill_path", "skills/clinical_agent_builder.md"),
            max_iterations=config_dict.get("max_iterations", 15),
            sandbox_n=config_dict.get("sandbox_n", 150),
            model=config_dict.get("model", "claude-sonnet-4-20250514"),
            agent_model=config_dict.get("agent_model",
                         config_dict.get("model", "claude-sonnet-4-20250514")),
            model_dir=config_dict.get("model_dir", "data/models"),
            max_concurrent=config_dict.get("max_concurrent", 10),
            use_batch_api=config_dict.get("use_batch_api", False),
            random_seed=config_dict.get("random_seed", 42),
        )

        def on_event(event_type: str, data: dict):
            """Callback from orchestrator to update state and emit events."""
            event_bus.emit(event_type, data)

            if event_type == "iteration_start":
                it = data.get("iteration", -1)
                run_state.current_iteration = it
                run_state.agent_specs[it] = {
                    "topology": data.get("topology"),
                    "agents": data.get("agents", []),
                    "rationale": data.get("rationale", ""),
                }
                run_state.step_status[it] = {
                    "build": "done", "run": "active",
                    "eval": "pending", "analyze": "pending", "update": "pending"
                }
                run_state.add_log("info",
                    f"Iteration {it} — "
                    f"Topology: {data.get('topology', '?')}, "
                    f"Agents: {data.get('agents', [])}", event_bus)

            elif event_type == "case_start":
                pid = data.get("patient_id", "?")
                run_state.add_log("debug",
                    f"Case {pid} — starting ({data.get('topology', '?')})", event_bus)

            elif event_type == "agent_start":
                pid = data.get("patient_id", "?")
                aid = data.get("agent_id", "?")
                run_state.add_log("debug",
                    f"  {pid} → agent '{aid}' started (role: {data.get('role', '?')}, tools: {data.get('n_tools', 0)})", event_bus)

            elif event_type == "agent_tool_call":
                pid = data.get("patient_id", "?")
                aid = data.get("agent_id", "?")
                tool = data.get("tool", "?")
                run_state.add_log("debug",
                    f"  {pid} → {aid} called {tool} (turn {data.get('turn', '?')})", event_bus)

            elif event_type == "agent_complete":
                pid = data.get("patient_id", "?")
                aid = data.get("agent_id", "?")
                tokens = data.get("tokens", 0)
                turns = data.get("turns", 0)
                has_score = data.get("has_risk_score", False)
                run_state.add_log("debug",
                    f"  {pid} → agent '{aid}' done ({turns} turns, {tokens} tok"
                    f"{', has score' if has_score else ''})", event_bus)

            elif event_type == "rate_limit_wait":
                aid = data.get("agent_id", "?")
                wait = data.get("wait_s", 0)
                run_state.add_log("warn",
                    f"Rate limit — waiting {wait}s (agent: {aid}, attempt {data.get('attempt', '?')})", event_bus)

            elif event_type == "api_overloaded":
                run_state.add_log("warn",
                    f"API overloaded — waiting {data.get('wait_s', 0)}s", event_bus)

            elif event_type == "case_complete":
                pid = data.get("patient_id", "?")
                correct = data.get("correct", False)
                score = data.get("risk_score", 0)
                gt = data.get("ground_truth", "?")
                tokens = data.get("total_tokens", 0)
                mark = "correct" if correct else "WRONG"
                run_state.add_log("info",
                    f"Case {pid} → {mark} (score={score:.3f}, truth={gt}, {tokens} tok, "
                    f"{data.get('wall_time_ms', 0)}ms)", event_bus)

            elif event_type == "case_progress":
                completed = data.get("completed", 0)
                total = data.get("total", 0)
                run_state.add_log("info",
                    f"Progress: {completed}/{total} cases complete", event_bus)

            elif event_type == "iteration_complete":
                it = data.get("iteration", -1)
                run_state.iterations.append(data)
                auc = data.get("metrics", {}).get("auc", 0)
                if auc > run_state.best_auc:
                    run_state.best_auc = auc
                    run_state.best_iter = it
                if it in run_state.step_status:
                    run_state.step_status[it]["run"] = "done"
                    run_state.step_status[it]["eval"] = "done"
                    run_state.step_status[it]["analyze"] = "active"
                run_state.add_log("info",
                    f"Iteration {it} complete — AUC: {auc:.4f}", event_bus)

            elif event_type == "analysis_start":
                it = data.get("iteration", -1)
                if it in run_state.step_status:
                    run_state.step_status[it]["analyze"] = "active"

            elif event_type == "analysis_complete":
                it = data.get("iteration", -1)
                run_state.analyses[it] = {
                    "findings": data.get("findings", []),
                    "evidence_classes": data.get("evidence_classes", []),
                }
                if it in run_state.step_status:
                    run_state.step_status[it]["analyze"] = "done"
                    run_state.step_status[it]["update"] = "active"
                n = len(data.get("findings", []))
                run_state.add_log("info",
                    f"Trace analysis: {n} findings", event_bus)

            elif event_type == "skill_update":
                it = data.get("iteration", -1)
                run_state.skill_updates[it] = data.get("diff_log", [])
                if it in run_state.step_status:
                    run_state.step_status[it]["update"] = "done"
                n = len(data.get("diff_log", []))
                run_state.add_log("info",
                    f"Skill updated: {n} edits applied", event_bus)

            elif event_type == "converged":
                run_state.add_log("info",
                    f"Converged: {data.get('reason', '')}", event_bus)

        run_state.add_log("info", "Loading patient data and ML models...", event_bus)

        results = await run_abstral(config, on_event=on_event)

        run_state.status = "completed"
        event_bus.emit("run_complete", {
            "total_iterations": len(results),
            "best_auc": run_state.best_auc,
            "best_iter": run_state.best_iter,
        })
        run_state.add_log("info",
            f"Run complete — {len(results)} iterations, "
            f"best AUC: {run_state.best_auc:.4f} at iter {run_state.best_iter}",
            event_bus)

    except asyncio.CancelledError:
        run_state.status = "stopped"
        run_state.add_log("warn", "Run cancelled", event_bus)

    except Exception as e:
        run_state.status = "error"
        run_state.error = str(e)
        event_bus.emit("run_error", {"error": str(e), "traceback": traceback.format_exc()})
        run_state.add_log("error", f"Run failed: {e}", event_bus)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="ABSTRAL Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8420, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    print(f"\n  ABSTRAL Dashboard → http://localhost:{args.port}\n")
    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
