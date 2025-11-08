# db.py (trecho mínimo para não quebrar)
import json, os
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).with_name("users.json")
LOG_PATH = Path(__file__).with_name("logs.json")

def _now(): return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def get_users():
    if not DB_PATH.exists(): return []
    try: return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except Exception: return []

def save_users(users):
    DB_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")

def add_user(name_or_dict, level=None, image_path=None):
    users = get_users()
    if isinstance(name_or_dict, dict):
        u = name_or_dict
    else:
        u = {"name": name_or_dict, "level": int(level or 1), "image_path": image_path}
    # evita duplicado simples
    users = [x for x in users if x.get("name") != u.get("name")]
    users.append(u)
    save_users(users)
    return True

def update_user_level(name, new_level):
    users = get_users()
    ok = False
    for u in users:
        if u.get("name") == name:
            u["level"] = int(new_level)
            ok = True
            break
    if ok: save_users(users)
    return ok

def update_user_image_path(name, path):
    users = get_users()
    ok = False
    for u in users:
        if u.get("name") == name:
            u["image_path"] = path
            ok = True
            break
    if ok: save_users(users)
    return ok

def get_logs():
    if not LOG_PATH.exists(): return []
    try: return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception: return []

def log_event(status, user_name=None, score=None, note=None):
    logs = get_logs()
    logs.append({
        "ts": _now(),
        "status": status,
        "user_name": user_name,
        "score": score,
        "note": note,
    })
    LOG_PATH.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
