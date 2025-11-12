# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from jinja2 import TemplateNotFound
import base64, io, inspect, os, time, secrets
from PIL import Image
import numpy as np
import cv2

# --- DB helpers ---
from db import get_users, add_user, get_logs, log_event, update_user_level
try:
    from db import update_user_image_path   # opcional
except ImportError:
    update_user_image_path = None

# --- Face helpers ---
from face_utils import save_face_image, predict_face, train_model, LBPH_THRESHOLD, load_label_map

app = Flask(__name__)
app.secret_key = "dev-secret-change-me-stronger-key"  # MUDE EM PRODUÇÃO
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # ok para dev local

# ---------------- Config ----------------
LEVEL_LABELS = {1: "Nível 1 (Geral)", 2: "Nível 2 (Diretoria)", 3: "Nível 3 (Ministro)"}
ADMIN_USER = "admin"
ADMIN_PASS = "0000"

# Requer liveness para login /api/verify
LIVENESS_REQUIRED = True
LIVENESS_WINDOW_SEC = 20

# ---------------- Utils ----------------
def b64_to_image(b64data: str):
    """Converte base64 (dataURL ou cru) para ndarray BGR (OpenCV)."""
    if "," in b64data:
        b64data = b64data.split(",", 1)[1]
    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)[:, :, ::-1]

def _repredict_once(img):
    """Treina e tenta predizer 1x novamente (para casos logo após cadastro)."""
    try:
        ok = train_model()
        if ok:
            time.sleep(0.1)
            return predict_face(img)
    except Exception as e:
        log_event(status="api_error", note=f"_repredict_once: {e}")
    return None, None, None

# ---------------- Páginas ----------------
@app.get("/")
def landing():
    return render_template("landing_dark.html")

@app.get("/validate")
def validate_view():
    return render_template("index.html", level_labels=LEVEL_LABELS, threshold=LBPH_THRESHOLD)

@app.get("/enroll")
def enroll_gate_redirect():
    return redirect(url_for("enroll_gate"))

@app.get("/enroll-gate")
def enroll_gate():
    return render_template("enroll_gate_dark.html", threshold=LBPH_THRESHOLD)

@app.get("/enroll-form")
def enroll_form():
    if session.get("user_level") == 3 or session.get("admin_ok"):
        return render_template("enroll.html", level_labels=LEVEL_LABELS)
    return redirect(url_for("enroll_gate"))

@app.get("/overview")
def overview():
    name = session.get("user_name")
    level = session.get("user_level", 1)
    if not name:
        log_event(status="access_denied", note="overview sem sessão")
        return redirect(url_for("landing"))
    return render_template("overview_dark.html", name=name, level=level, level_label=LEVEL_LABELS.get(level))

@app.get("/home")
def home_view():
    name = session.get("user_name")
    level = session.get("user_level", 1)
    if not name:
        log_event(status="access_denied", note="home sem sessão")
        return redirect(url_for("landing"))
    return render_template("home_dark.html", name=name, level=level, level_label=LEVEL_LABELS.get(level))

# ---------------- Pesquisas ----------------
RESEARCH = [
    {"slug":"saxitoxina","titulo":"Saxitoxina","categoria":"Toxinas Naturais","origem":"Microalgas marinhas (maré vermelha)","risco":"Bloqueia canais de sódio; paralisia e insuficiência respiratória em minutos; alta letalidade","aplicacao":"Estudos neurofisiológicos e transmissão sináptica","summary":"Neurotoxina marinha com rápida ação neuromuscular e alto risco; foco de pesquisa em canais iônicos","report":"reports/saxitoxina.pdf"},
    {"slug":"tetrodotoxina","titulo":"Tetrodotoxina","categoria":"Toxinas Naturais","origem":"Baiacu e outros animais marinhos","risco":"~1.200× mais tóxica que o cianeto; risco de parada cardiorrespiratória","aplicacao":"Pesquisa em anestésicos e bloqueadores de canais iônicos","summary":"Altíssima potência; usada como ferramenta em fisiologia neuronal e analgesia experimental","report":"reports/tetrodotoxina.pdf"},
    {"slug":"ricina","titulo":"Ricina","categoria":"Toxinas Naturais","origem":"Sementes de mamona (Ricinus communis)","risco":"Inibe ribossomos; dose letal ~0,2 mg/kg; potencial uso como agente biológico","aplicacao":"Imunotoxinas para oncologia (pesquisa)","summary":"Toxina proteica que bloqueia síntese proteica; investigada como componente terapêutico experimental","report":"reports/ricina.pdf"},
    {"slug":"dioxinas","titulo":"Dioxinas","categoria":"Agentes Químicos Industriais","origem":"Subprodutos de processos industriais e incineração","risco":"Altamente carcinogênicas e teratogênicas; bioacumulativas","aplicacao":"Sem uso benéfico; foco em contenção e monitoramento","summary":"Contaminantes persistentes de alto impacto ambiental e à saúde; gestão ambiental rigorosa","report":"reports/dioxinas.pdf"},
    {"slug":"mercurio-metilado","titulo":"Mercúrio Metilado","categoria":"Agentes Químicos Industriais","origem":"Metilação de mercúrio inorgânico em ecossistemas aquáticos","risco":"Neurotoxina persistente; bioacumula em peixes e frutos do mar","aplicacao":"Uso extremamente controlado em pesquisa de catalisadores","summary":"Exige políticas de controle e vigilância por impactos neurológicos e bioacumulação","report":"reports/mercurio-metilado.pdf"},
    {"slug":"nicotinoides-sinteticos","titulo":"Nicotinoides Sintéticos","categoria":"Agentes Químicos Industriais","origem":"Inseticidas agrícolas","risco":"Associados ao colapso de polinizadores; impacto ecossistêmico","aplicacao":"Proteção de cultivos em contextos controlados","summary":"Risco a abelhas e biodiversidade; regulamentação e boas práticas são essenciais","report":"reports/nicotinoides-sinteticos.pdf"},
    {"slug":"h5n1-modificado","titulo":"H5N1 Modificado","categoria":"Patógenos Modificados","origem":"Variante de influenza aviária alterada em laboratório","risco":"Potencial pandêmico com alta letalidade; aumento de transmissibilidade","aplicacao":"Desenvolvimento de vacinas e estudos imunológicos","summary":"Pesquisa de alto controle para vigilância e preparação a pandemias","report":"reports/h5n1-modificado.pdf"},
    {"slug":"bacillus-anthracis-resistente","titulo":"Bacillus anthracis Resistente","categoria":"Patógenos Modificados","origem":"Cepa de antraz com resistência múltipla","risco":"Elevado risco de uso indevido/bioterrorismo","aplicacao":"P&D de antídotos e vacinas","summary":"Necessita biossegurança máxima e rastreabilidade","report":"reports/bacillus-anthracis-resistente.pdf"},
    {"slug":"candida-auris-termotolerante","titulo":"Candida auris Termotolerante","categoria":"Patógenos Modificados","origem":"Fungo hospitalar emergente tolerante a altas temperaturas","risco":"Surtos hospitalares; resistência a antifúngicos","aplicacao":"Pesquisa de mecanismos de resistência e novas terapias","summary":"Patógeno desafiador para controle infeccioso; exige vigilância","report":"reports/candida-auris-termotolerante.pdf"},
    {"slug":"biologia-sintetica","titulo":"Biologia Sintética","categoria":"Tema Sensível","origem":"Engenharia genética avançada","risco":"Criação de organismos patogênicos/artificiais; riscos de dual use","aplicacao":"Terapias gênicas, bioengenharia segura e bioremediação","summary":"Potencial transformador com riscos regulatórios e éticos","report":"reports/biologia-sintetica.pdf"}
]

@app.get("/pesquisas")
def pesquisas_list():
    level = session.get("user_level", 1)
    if level < 2:
        log_event(status="access_denied", user_name=session.get("user_name"),
                  note="Tentativa de acessar pesquisas sem Nível 2+")
        return redirect(url_for("overview"))
    return render_template("pesquisas_list_dark.html", level=level,
                           level_label=LEVEL_LABELS.get(level), docs=RESEARCH)

@app.get("/pesquisas/<slug>")
def pesquisa_detail(slug):
    level = session.get("user_level", 1)
    if level < 2:
        log_event(status="access_denied", user_name=session.get("user_name"),
                  note=f"Tentativa de acessar pesquisa {slug} sem Nível 2+")
        return redirect(url_for("overview"))
    doc = next((d for d in RESEARCH if d["slug"] == slug), None)
    if not doc:
        log_event(status="not_found", user_name=session.get("user_name"),
                  note=f"Pesquisa {slug} não encontrada")
        return redirect(url_for("pesquisas_list"))
    try:
        return render_template(f"reports/{slug}.html", level=level,
                               level_label=LEVEL_LABELS.get(level), doc=doc)
    except TemplateNotFound:
        return render_template("pesquisa_detail_dark.html", level=level,
                               level_label=LEVEL_LABELS.get(level), doc=doc)

# ---------------- Gate de cadastro (N3/Admin) ----------------
@app.post("/api/verify_enroll")
def api_verify_enroll():
    """
    Gate de matrícula:
    - N3 reconhecido passa direto ao formulário.
    - Outros níveis: precisa admin.
    - Não reconhecido: precisa admin.
    (Liveness AQUI é opcional; geralmente exigimos só no login)
    """
    data = request.get_json(force=True) or {}
    img64 = (data.get("image_b64") or "").strip()
    if not img64:
        log_event(status="api_error", note="api_verify_enroll: Imagem não enviada.")
        return jsonify({"ok": False, "error": "Imagem não enviada."}), 400

    try:
        img = b64_to_image(img64)
    except Exception as e:
        log_event(status="api_error", note=f"api_verify_enroll: b64_to_image falhou: {e}")
        return jsonify({"ok": False, "error": "Imagem inválida."}), 400

    try:
        label, conf, bbox = predict_face(img)
    except Exception as e:
        log_event(status="api_error", note=f"api_verify_enroll: predict_face falhou: {e}")
        return jsonify({"ok": False, "error": "Erro no processamento da imagem."}), 500

    if label is None:
        log_event(status="enroll_gate_failed", note="api_verify_enroll: Rosto não detectado / modelo vazio")
        return jsonify({"ok": True, "match": False, "reason": "Rosto não detectado",
                        "require_admin": True, "bbox": bbox}), 200

    try:
        conf_val = float(conf) if conf is not None else 1e9
        thr_val = float(LBPH_THRESHOLD)
    except Exception:
        conf_val, thr_val = 1e9, 70.0

    if conf_val <= thr_val:
        try:
            lm = load_label_map()
            name = lm.get(label, "Usuário reconhecido")
        except Exception:
            name = "Usuário reconhecido"
        try:
            users = get_users()
            user = next((u for u in users if u.get("name") == name), None)
            level = int(user.get("level", 1)) if user else 1
        except Exception:
            level = 1

        level_label = LEVEL_LABELS.get(level, f"Nível {level}")
        log_event(status="enroll_gate_face", user_name=name, score=float(conf_val), note=level_label)

        if level == 3:
            session["user_name"] = name
            session["user_level"] = level
            return jsonify({"ok": True, "match": True, "is_n3": True, "name": name,
                            "level": level, "bbox": bbox, "redirect": url_for("enroll_form")}), 200

        return jsonify({"ok": True, "match": True, "is_n3": False, "name": name,
                        "level": level, "bbox": bbox, "require_admin": True}), 200

    log_event(status="enroll_gate_failed", score=float(conf_val),
              note="api_verify_enroll: Conf acima do threshold")
    return jsonify({"ok": True, "match": False, "reason": "Sem correspondência",
                    "require_admin": True, "bbox": bbox}), 200

@app.post("/admin-login")
def admin_login():
    data = request.get_json(force=True)
    user = data.get("user", "").strip()
    pw = data.get("password", "").strip()

    if user == ADMIN_USER and pw == ADMIN_PASS:
        session["admin_ok"] = True
        log_event(status="admin_login_ok", user_name="admin")
        return jsonify({"ok": True, "redirect": url_for("enroll_form")})
    log_event(status="admin_login_failed", note="Credenciais inválidas")
    return jsonify({"ok": False, "error": "Credenciais inválidas"}), 401

# ---------------- Cadastro ----------------
@app.post("/api/enroll")
def api_enroll():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    try:
        level = int(data.get("level", 1))
    except ValueError:
        level = 1
        log_event(status="api_error", note=f"api_enroll: Nível inválido ({data.get('level')})")

    img64 = data.get("image_b64")
    if not name or not img64:
        log_event(status="api_error", note="api_enroll: nome/imagem ausentes")
        return jsonify({"ok": False, "error": "Nome e imagem são obrigatórios."}), 400

    if not (session.get("user_level") == 3 or session.get("admin_ok")):
        log_event(status="access_denied", user_name=session.get("user_name"),
                  note="api_enroll: sem permissão")
        return jsonify({"ok": False, "error": "Somente Nível 3 ou Admin podem cadastrar/editar."}), 403

    img = b64_to_image(img64)

    # tenta reconhecer para atualizar
    label, conf, bbox = None, None, None
    try:
        label, conf, bbox = predict_face(img)
    except Exception as e:
        log_event(status="api_error", note=f"api_enroll: predict_face: {e}")

    if label is not None and conf is not None and float(conf) <= float(LBPH_THRESHOLD):
        lm = load_label_map()
        current_name = lm.get(label)
        if current_name:
            updated_level_flag = False
            try:
                updated_level_flag = bool(update_user_level(current_name, level))
            except Exception as e:
                log_event(status="api_error", user_name=current_name,
                          note=f"api_enroll: update_user_level: {e}")

            saved_sample_flag = False
            try:
                path_rel_update = save_face_image(current_name, img, level)
                if path_rel_update:
                    saved_sample_flag = True
                    if callable(update_user_image_path):
                        try:
                            update_user_image_path(current_name, path_rel_update)
                        except Exception as e:
                            log_event(status="api_error", user_name=current_name,
                                      note=f"api_enroll: update_user_image_path: {e}")
            except Exception as e:
                log_event(status="api_error", user_name=current_name,
                          note=f"api_enroll: salvar amostra: {e}")

            try:
                ok = train_model()
                if ok:
                    time.sleep(0.15)
            except Exception as e:
                log_event(status="api_error", note=f"api_enroll: treinar após update: {e}")

            log_event(status="enroll_update_level", user_name=current_name,
                      note=f"Nível->{level}; update_level={updated_level_flag}; nova_amostra={saved_sample_flag}")
            return jsonify({
                "ok": True, "updated_only": True, "name": current_name,
                "level": level, "saved_sample": saved_sample_flag,
                "note": "Nível/amostra atualizados para rosto já cadastrado."
            })

    # novo cadastro
    path_rel = save_face_image(name, img, level)
    if not path_rel:
        log_event(status="enroll_failed", user_name=name, note="Rosto não detectado")
        return jsonify({"ok": False, "error": "Rosto não detectado na imagem."}), 200

    try:
        sig = inspect.signature(add_user)
        if len(sig.parameters) == 3:
            add_user(name, level, path_rel)
        else:
            add_user({"name": name, "level": level, "image_path": path_rel})
    except Exception as e:
        log_event(status="enroll_failed_db", user_name=name, note=f"add_user: {e}")
        return jsonify({"ok": False, "error": f"Falha ao salvar no DB: {e}"}), 500

    try:
        ok = train_model()
        if ok:
            time.sleep(0.15)  # flush para o próximo predict já ver o modelo novo
    except Exception as e:
        log_event(status="api_error", note=f"api_enroll: treinar após novo cadastro: {e}")

    log_event(status="enroll_ok", user_name=name, note=f"Novo usuário nível {level}")
    return jsonify({"ok": True, "note": "Novo usuário cadastrado com sucesso."})

# ---------------- Liveness ----------------
@app.post("/api/liveness_challenge")
def liveness_challenge():
    actions_pool = ["pisque os olhos", "vire o rosto para a esquerda", "vire o rosto para a direita", "abra a boca", "sorria"]
    actions = secrets.choice(actions_pool), secrets.choice(actions_pool)
    nonce = secrets.token_urlsafe(16)

    session["live_nonce"] = nonce
    session["live_issued_at"] = time.time()
    session["live_ok"] = False

    return jsonify({"ok": True, "nonce": nonce, "actions": list(actions)})

def _b64_to_bgr(b64data: str):
    if "," in b64data:
        b64data = b64data.split(",", 1)[1]
    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)[:, :, ::-1]

def _optical_flow_score(frames_bgr):
    if len(frames_bgr) < 3: return 0.0
    mags = []
    prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames_bgr)):
        nxt = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append(float(np.median(mag)))
        prev = nxt
    return float(np.mean(mags)) if mags else 0.0

def _glare_ratio(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float((gray > 245).mean())

def _blur_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

@app.post("/api/liveness_complete")
def liveness_complete():
    data = request.get_json(force=True)
    client_nonce = data.get("nonce", "")
    frames_b64 = data.get("frames", [])

    # 1) validação do nonce/janela
    if client_nonce != session.get("live_nonce"):
        return jsonify({"ok": False, "error": "nonce inválido"}), 400
    if (time.time() - session.get("live_issued_at", 0)) > LIVENESS_WINDOW_SEC:
        return jsonify({"ok": False, "error": "desafio expirado"}), 400

    # 2) frames -> movimento
    frames_bgr = []
    for b64 in frames_b64[:12]:
        try:
            frames_bgr.append(_b64_to_bgr(b64))
        except Exception:
            pass
    if len(frames_bgr) < 3:
        return jsonify({"ok": False, "error": "frames insuficientes"}), 400

    flow = _optical_flow_score(frames_bgr)
    mid = frames_bgr[len(frames_bgr)//2]
    glare = _glare_ratio(mid)
    blurv = _blur_score(mid)

    MIN_FLOW, MAX_GLARE, MIN_BLUR_V = 0.50, 0.25, 30.0
    if flow < MIN_FLOW:
        return jsonify({"ok": False, "error": "pouca variação temporal"}), 200
    if glare > MAX_GLARE:
        return jsonify({"ok": False, "error": "brilho/saturação excessivos (tela?)"}), 200
    if blurv < MIN_BLUR_V:
        return jsonify({"ok": False, "error": "imagem artificial/desfocada"}), 200

    session["live_ok"] = True
    session["live_valid_until"] = time.time() + LIVENESS_WINDOW_SEC
    return jsonify({"ok": True})

# ---------------- Login ----------------
@app.post("/api/verify")
def api_verify():
    """
    Autenticação facial (login):
    - Se LIVENESS_REQUIRED=True: bloqueia com 409 quando faltando/expirado (NÃO consome aqui).
    - Predição LBPH; se falhar, re-treina e tenta 1x.
    - Heurísticas anti-foto simples.
    - Consome o token de liveness APÓS sucesso.
    """
    # Liveness obrigatório?
    if LIVENESS_REQUIRED:
        import time as _time
        live_ok = bool(session.get("live_ok", False))
        live_until = float(session.get("live_valid_until", 0))
        if not (live_ok and _time.time() <= live_until):
            log_event(status="auth_blocked", note="api_verify: liveness ausente/expirado")
            return jsonify({
                "ok": False,
                "error": "Faça a verificação de vivacidade antes do login.",
                "require_liveness": True
            }), 409

    data = request.get_json(force=True) or {}
    img64 = (data.get("image_b64") or "").strip()
    if not img64:
        log_event(status="api_error", note="api_verify: Imagem não enviada.")
        return jsonify({"ok": False, "error": "Imagem não enviada."}), 400

    # Converte imagem
    try:
        img = b64_to_image(img64)
    except Exception as e:
        log_event(status="api_error", note=f"api_verify: b64_to_image falhou: {e}")
        return jsonify({"ok": False, "error": "Imagem inválida."}), 400

    # 1ª predição
    try:
        label, conf, bbox = predict_face(img)
    except Exception as e:
        log_event(status="api_error", note=f"api_verify: predict_face falhou: {e}")
        return jsonify({"ok": False, "error": "Erro no processamento da imagem."}), 500

    # Sem rosto/modelo: tenta re-treinar e prever 1x
    if label is None:
        log_event(status="auth_warn", note="api_verify: label=None; tentando repredict")
        label, conf, bbox = _repredict_once(img)
        if label is None:
            return jsonify({"ok": True, "match": False,
                            "reason": "Rosto não detectado ou modelo vazio",
                            "bbox": bbox}), 200

    # Heurísticas anti-foto
    try:
        H, W = img.shape[:2]
        if bbox:
            x, y, w, h = bbox
            area_ratio = (w * h) / float(W * H + 1e-6)
            if area_ratio < 0.04 or area_ratio > 0.75:
                log_event(status="auth_failed", note=f"api_verify: área rosto fora do esperado ({area_ratio:.3f})")
                return jsonify({"ok": True, "match": False,
                                "reason": "Rosto fora do enquadramento esperado",
                                "bbox": bbox}), 200
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[max(0,y):y+h, max(0,x):x+w]
            if roi.size > 0:
                lap = cv2.Laplacian(roi, cv2.CV_64F).var()
                if lap < 25:
                    log_event(status="auth_failed", note=f"api_verify: nitidez muito baixa (Laplacian={lap:.1f})")
                    return jsonify({"ok": True, "match": False,
                                    "reason": "Imagem muito borrada/estática",
                                    "bbox": bbox}), 200
    except Exception:
        pass

    # Threshold
    try:
        conf_val = float(conf) if conf is not None else 1e9
        thr_val = float(LBPH_THRESHOLD)
    except Exception:
        conf_val, thr_val = 1e9, 70.0

    # Se acima do limiar, tenta repredict final
    if conf_val > thr_val:
        log_event(status="auth_warn", note=f"api_verify: acima do threshold ({conf_val:.2f}>{thr_val:.2f}); repredict")
        label2, conf2, bbox2 = _repredict_once(img)
        if label2 is not None:
            label, conf, bbox = label2, conf2, (bbox2 or bbox)
            try:
                conf_val = float(conf) if conf is not None else conf_val
            except Exception:
                pass

    if conf_val <= thr_val:
        try:
            lm = load_label_map()
            name = lm.get(label, "Usuário reconhecido")
        except Exception:
            name = "Usuário reconhecido"

        try:
            users = get_users()
            user = next((u for u in users if u.get("name") == name), None)
            level = int(user.get("level", 1)) if user else 1
        except Exception:
            level = 1

        level_label = LEVEL_LABELS.get(level, f"Nível {level}")
        session["user_name"] = name
        session["user_level"] = level

        # Consome o token de liveness SOMENTE após sucesso
        try:
            session["live_ok"] = False
        except Exception:
            pass

        log_event(status="auth_ok", user_name=name, score=float(conf_val), note=level_label)
        return jsonify({
            "ok": True,
            "match": True,
            "name": name,
            "level": level,
            "level_label": level_label,
            "distance": float(conf_val),
            "bbox": bbox,
            "redirect": url_for("overview")
        }), 200

    log_event(status="auth_failed", score=float(conf_val),
              note="api_verify: Conf acima do threshold (após repredict)")
    return jsonify({"ok": True, "match": False, "reason": "Sem correspondência", "bbox": bbox}), 200

# ---------------- Diagnóstico ----------------
@app.get("/api/model_status")
def model_status():
    from face_utils import MODEL_PATH, LABELS_PATH
    users = get_users()
    exists_model = os.path.exists(MODEL_PATH)
    exists_labels = os.path.exists(LABELS_PATH)
    return jsonify({
        "ok": True,
        "users_count": len(users),
        "model_exists": exists_model,
        "labels_exists": exists_labels,
        "threshold": float(LBPH_THRESHOLD)
    })

@app.post("/api/retrain")
def api_retrain():
    try:
        ok = train_model()
        if ok:
            time.sleep(0.15)
        return jsonify({"ok": bool(ok)})
    except Exception as e:
        log_event(status="api_error", note=f"api_retrain: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------- Sessão ----------------
@app.get("/logout")
def logout():
    session.clear()
    log_event(status="logout", note="Usuário deslogado.")
    return redirect(url_for("landing"))

# ---------------- Main ----------------
if __name__ == "__main__":
    # Para HTTPS em rede local, gere certs e use ssl_context
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
