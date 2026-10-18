from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from app.admin import esc, layout, require_admin
from app.store import audit, connect, list_inbound_series, list_return_jobs, setting, set_setting

router = APIRouter(prefix="/admin", tags=["operations"])


@router.get("/queues", response_class=HTMLResponse)
def queues(_: str = Depends(require_admin)):
    inbound = "".join(f"<tr><td>{esc(x['series_id'])}</td><td>{esc(x['center_name'] or x['source_aet'])}</td><td>{esc(x['description'])}</td><td>{x['instances']}</td><td><span class='badge warn'>{esc(x['status'])}</span></td><td>{esc(x['error'])}</td></tr>" for x in list_inbound_series(300)) or "<tr><td colspan='6' class='empty'>Empty</td></tr>"
    with connect() as db:
        jobs = db.execute("SELECT * FROM comparison_jobs WHERE archived=0 ORDER BY id DESC LIMIT 300").fetchall()
    enhancement = "".join(f"<tr><td>{j['id']}</td><td>{esc(j['profile'])}/{esc(j['strength'])}</td><td>{j['processed_instances']}/{j['total_instances']}</td><td><span class='badge {'ok' if j['status']=='ready' else 'bad' if j['status']=='failed' else 'warn'}'>{esc(j['status'])}</span></td><td>{esc(j['qc_status'])}</td><td>{esc(j['error'])}</td><td>{f'''<form method='post' action='/admin/queues/enhancement/{j['id']}/cancel'><button class='btn red'>Cancel</button></form>''' if j['status']=='queued' else ''}</td></tr>" for j in jobs) or "<tr><td colspan='7' class='empty'>Empty</td></tr>"
    returns = "".join(f"<tr><td>{r['id']}</td><td>{esc(r['center_name'])}</td><td class='route'>{esc(r['return_aet'])}@{esc(r['return_ip'])}:{r['return_port']}</td><td>{r['attempts']}</td><td><span class='badge {'ok' if r['status']=='sent' else 'bad' if r['status']=='failed' else 'warn'}'>{esc(r['status'])}</span></td><td>{esc(r['error'])}</td><td>{f'''<form method='post' action='/admin/returns/{r['id']}/retry'><button class='btn gray'>Retry</button></form>''' if r['status']=='failed' else ''}</td></tr>" for r in list_return_jobs(300)) or "<tr><td colspan='7' class='empty'>Empty</td></tr>"
    body = f"""<div class='actions'><a class='btn gray' href='/admin'>Dashboard</a><a class='btn gray' href='/admin/services'>Services</a></div>
    <section class='panel'><div class='panel-head'><h2>Receive Queue</h2><form method='post' action='/admin/queues/receive/archive-completed' onsubmit="return confirm('Archive completed receive records?')"><button class='btn gray'>Clear completed</button></form></div><div class='table-wrap'><table><tr><th>Series</th><th>Branch</th><th>Description</th><th>Images</th><th>Status</th><th>Error</th></tr>{inbound}</table></div></section>
    <section class='panel'><div class='panel-head'><h2>Enhancement Queue</h2><form method='post' action='/admin/queues/enhancement/archive-completed' onsubmit="return confirm('Archive completed enhancement jobs?')"><button class='btn gray'>Clear completed</button></form></div><div class='table-wrap'><table><tr><th>Job</th><th>Profile</th><th>Progress</th><th>Status</th><th>QC</th><th>Error</th><th>Action</th></tr>{enhancement}</table></div></section>
    <section class='panel'><div class='panel-head'><h2>Return Queue</h2><form method='post' action='/admin/queues/return/archive-completed' onsubmit="return confirm('Archive sent return records?')"><button class='btn gray'>Clear sent</button></form></div><div class='table-wrap'><table><tr><th>ID</th><th>Branch</th><th>Destination</th><th>Attempts</th><th>Status</th><th>Error</th><th>Action</th></tr>{returns}</table></div></section>"""
    return layout("Queue Management", body)


@router.post("/queues/enhancement/{job_id}/cancel")
def cancel_job(job_id: int, _: str = Depends(require_admin)):
    with connect() as db:
        result = db.execute("UPDATE comparison_jobs SET status='cancelled',updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='queued'", (job_id,))
    if result.rowcount != 1:
        raise HTTPException(409, "Only queued jobs can be cancelled")
    audit("JOB_CANCELLED", "comparison_job", job_id, actor=_)
    return RedirectResponse("/admin/queues", 303)


@router.post("/queues/{queue}/archive-completed")
def archive_completed(queue: str, _: str = Depends(require_admin)):
    rules = {
        "receive": ("inbound_series", "status IN ('READY','SENT','FAILED','RETURN_FAILED')"),
        "enhancement": ("comparison_jobs", "status IN ('ready','failed','cancelled')"),
        "return": ("return_jobs", "status='sent'"),
    }
    if queue not in rules:
        raise HTTPException(404, "Unknown queue")
    table, condition = rules[queue]
    with connect() as db:
        result = db.execute(f"UPDATE {table} SET archived=1 WHERE archived=0 AND {condition}")
    audit("QUEUE_ARCHIVED", queue, "completed", f"count={result.rowcount}", actor=_)
    return RedirectResponse("/admin/queues?message=" + quote(f"Archived {result.rowcount} records"), 303)


@router.get("/services", response_class=HTMLResponse)
def services(_: str = Depends(require_admin)):
    items = [("intake", "Receive / Intake Worker"), ("enhancement", "Enhancement Worker"), ("return", "Return Worker")]
    rows = "".join(f"<tr><td><b>{label}</b></td><td><span class='badge {'warn' if setting('pause_'+key)=='1' else 'ok'}'>{'PAUSED' if setting('pause_'+key)=='1' else 'RUNNING'}</span></td><td><form method='post' action='/admin/services/{key}/toggle'><button class='btn gray'>{'Resume' if setting('pause_'+key)=='1' else 'Pause'}</button></form></td></tr>" for key,label in items)
    maintenance = setting("maintenance_mode") == "1"
    body = f"""<div class='actions'><a class='btn gray' href='/admin'>Dashboard</a><a class='btn gray' href='/admin/queues'>Queues</a></div><section class='panel'><div class='panel-head'><h2>Service Control</h2><span class='badge {'bad' if maintenance else 'ok'}'>{'MAINTENANCE' if maintenance else 'NORMAL'}</span></div><div class='notice'>Safe logical controls only. Docker/Nginx restart requires the separate host Control Agent and is not exposed through the application container.</div><form method='post' action='/admin/services/maintenance/toggle'><button class='btn {'red' if not maintenance else 'gray'}'>{'Enable Maintenance Mode' if not maintenance else 'Disable Maintenance Mode'}</button></form><div class='table-wrap'><table><tr><th>Worker</th><th>Status</th><th>Control</th></tr>{rows}</table></div></section>"""
    return layout("Service Control", body)


@router.post("/services/{worker}/toggle")
def toggle_worker(worker: str, _: str = Depends(require_admin)):
    if worker == "maintenance":
        key = "maintenance_mode"
    elif worker in {"intake", "enhancement", "return"}:
        key = "pause_" + worker
    else:
        raise HTTPException(404, "Unknown worker")
    value = "0" if setting(key) == "1" else "1"
    set_setting(key, value)
    audit("SETTING_CHANGED", "system_setting", key, f"value={value}", actor=_)
    return RedirectResponse("/admin/services", 303)
