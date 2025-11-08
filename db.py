# db.py
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_PATH = Path(__file__).with_name("users.json")
LOG_PATH = Path(__file__).with_name("logs.json")


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ---------- utilidades internas ----------

def _read_json(path: Path) -> Any:
    if not path.exists():
        return [] if path is DB_PATH else []
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return [] if path is DB_PATH else []
        return json.loads(raw)
    except Exception:
        # Em caso de corrupção, não derruba o app:
        return [] if path is DB_PATH else []


def _write_json(path: Path, data: Any) -> None:
    # Grava de forma simples e consistente
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_name(name: str) -> str:
    return (name or "").strip().lower()


# ---------- API de usuários ----------

def get_users() -> List[Dict[str, Any]]:
    data = _read_json(DB_PATH)
    return data if isinstance(data, list) else []


def save_users(users: List[Dict[str, Any]]) -> None:
    _write_json(DB_PATH, users)


def get_user_by_name(name: str) -> Optional[Dict[str, Any]]:
    target = _norm_name(name)
    for u in get_users():
        if _norm_name(u.get("name", "")) == target:
            return u
    return None


def set_user(name: str, level: int, image_path: str) -> bool:
    """
    Upsert por 'name' (case-insensitive).
    Se existir, atualiza level e image_path (se fornecidos).
    Se não existir, cria.
    Retorna True se houve mudança em disco.
    """
    users = get_users()
    target = _norm_name(name)
    changed = False
    found = False

    for u in users:
        if _norm_name(u.get("name", "")) == target:
            found = True
            # atualiza somente se diferente
            new_level = int(level or u.get("level", 1))
            new_path = (image_path or u.get("image_path", "")).strip()
            if int(u.get("level", 1)) != new_level:
                u["level"] = new_level
                changed = True
            if new_path and str(u.get("image_path", "")).strip() != new_path:
                u["image_path"] = new_path
                changed = True
            break

    if not found:
        users.append({
            "name": str(name).strip(),
            "level": int(level or 1),
            "image_path": str(image_path or "").strip()
        })
        changed = True

    if changed:
        save_users(users)
    return changed


def add_user(name_or_dict, level=None, image_path=None) -> bool:
    """
    Compatível com chamadas antigas:
      add_user("Nome", 2, "faces/Nome_L2_....png")
      add_user({"name":"Nome","level":2,"image_path":"faces/...png"})
    Agora faz *upsert* por nome (evita duplicar).
    """
    if isinstance(name_or_dict, dict):
        nm = str(name_or_dict.get("name", "")).strip()
        lv = int(name_or_dict.get("level", 1))
        ip = str(name_or_dict.get("image_path", "")).strip()
        return set_user(nm, lv, ip)
    else:
        nm = str(name_or_dict).strip()
        lv = int(level or 1)
        ip = str(image_path or "").strip()
        return set_user(nm, lv, ip)


def update_user_level(name: str, new_level: int) -> bool:
    """Atualiza o nível por nome (case-insensitive). Retorna True se mudou algo."""
    users = get_users()
    target = _norm_name(name)
    changed = False
    for u in users:
        if _norm_name(u.get("name", "")) == target:
            if int(u.get("level", 1)) != int(new_level):
                u["level"] = int(new_level)
                changed = True
            break
    if changed:
        save_users(users)
    return changed


def update_user_image_path(name: str, new_rel_path: str) -> bool:
    """Atualiza image_path do usuário (por nome). Retorna True se mudou algo."""
    users = get_users()
    target = _norm_name(name)
    changed = False
    for u in users:
        if _norm_name(u.get("name", "")) == target:
            if str(u.get("image_path", "")).strip() != str(new_rel_path).strip():
                u["image_path"] = str(new_rel_path).strip()
                changed = True
            break
    if changed:
        save_users(users)
    return changed


# ---------- logs ----------

def get_logs() -> List[Dict[str, Any]]:
    data = _read_json(LOG_PATH)
    return data if isinstance(data, list) else []


def log_event(status: str, user_name: Optional[str] = None, score: Optional[float] = None, note: Optional[str] = None) -> None:
    logs = get_logs()
    logs.append({
        "ts": _now(),
        "status": status,
        "user_name": user_name,
        "score": score,
        "note": note
    })
    _write_json(LOG_PATH, logs)
