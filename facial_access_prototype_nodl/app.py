# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from jinja2 import TemplateNotFound
import base64, io, inspect, os
from pathlib import Path
from PIL import Image
import numpy as np

# ---- Caminhos base / templates / estáticos ----
BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
FACES_DIR = BASE_DIR / "faces"
MODELS_DIR = BASE_DIR / "models"

# Cria diretórios graváveis (filesystem do Render é efêmero, mas gravável)
FACES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- App Flask configurada para produção no Render ----
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me-stronger-key")
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # deixe False no free-tier (sem HTTPS direto)

LEVEL_LABELS = {1: "Nível 1 (Geral)", 2: "Nível 2 (Diretoria)", 3: "Nível 3 (Ministro)"}

# Credenciais de Administrador (APENAS PARA DESENVOLVIMENTO - MUDAR EM PRODUÇÃO)
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "0000")

# ---- Módulos auxiliares ----
from db import get_users, add_user, get_logs, log_event, update_user_level
try:
    from db import update_user_image_path  # opcional
except Exception:
    update_user_image_path = None

# Face utils
from face_utils import save_face_image, predict_face, train_model, LBPH_THRESHOLD, load_label_map

def b64_to_image(b64data: str):
    """Converte base64 para array numpy BGR (OpenCV)."""
    if "," in b64data:
        b64data = b64data.split(",", 1)[1]
    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)[:, :, ::-1]

# ------------------ ROTAS DE SAÚDE ------------------
@app.get("/health")
def health():
    return jsonify(status="ok"), 200

# ------------------ ROTAS DE PÁGINAS (FRONTEND) ------------------
@app.get("/")
def landing():
    """Página de entrada/landing page."""
    try:
        return render_template("landing_dark.html")
    except TemplateNotFound:
        # Fallback amigável para validar em produção caso o template não tenha sido enviado
        return (
            "<h1>Landing</h1><p>Template <code>landing_dark.html</code> não encontrado."
            " Confirme a pasta <code>templates/</code> no deploy.</p>", 200
        )

@app.get("/validate")
def validate_view():
    """Tela de validação facial para autenticação de usuários."""
    return render_template("index.html", level_labels=LEVEL_LABELS, threshold=LBPH_THRESHOLD)

@app.get("/enroll")
def enroll_gate_redirect():
    """Redireciona para o portão de cadastro. Útil para links externos."""
    return redirect(url_for("enroll_gate"))

@app.get("/enroll-gate")
def enroll_gate():
    """Portão de acesso para o formulário de cadastro/edição de usuários."""
    return render_template("enroll_gate_dark.html", threshold=LBPH_THRESHOLD)

@app.get("/enroll-form")
def enroll_form():
    """Formulário para cadastrar novo rosto ou atualizar usuário existente (N3/Admin)."""
    if session.get("user_level") == 3 or session.get("admin_ok"):
        return render_template("enroll.html", level_labels=LEVEL_LABELS)
    return redirect(url_for("enroll_gate"))

@app.get("/overview")
def overview():
    """Dashboard de visão geral após login."""
    name = session.get("user_name")
    level = session.get("user_level", 1)
    if not name:
        log_event(status="access_denied", note="Tentativa de acessar overview sem sessão")
        return redirect(url_for("landing"))
    return render_template("overview_dark.html", name=name, level=level, level_label=LEVEL_LABELS.get(level))

@app.get("/home")
def home_view():
    """Página inicial personalizada após login."""
    name = session.get("user_name")
    level = session.get("user_level", 1)
    if not name:
        log_event(status="access_denied", note="Tentativa de acessar home sem sessão")
        return redirect(url_for("landing"))
    return render_template("home_dark.html", name=name, level=level, level_label=LEVEL_LABELS.get(level))

# ------------------ DADOS DE PESQUISA ------------------
RESEARCH = [
    {
        "slug": "saxitoxina",
        "titulo": "Saxitoxina",
        "categoria": "Toxinas Naturais",
        "origem": "Microalgas marinhas (maré vermelha)",
        "risco": "Bloqueia canais de sódio; paralisia e insuficiência respiratória em minutos; alta letalidade",
        "aplicacao": "Estudos neurofisiológicos e transmissão sináptica",
        "summary": "Neurotoxina marinha com rápida ação neuromuscular e alto risco; foco de pesquisa em canais iônicos.",
        "autor_lider": "Dra. Ana Paula Silva",
        "data_publicacao": "2023-08-15",
        "metodologia": "Estudos in vitro e in vivo; análise cromatográfica.",
        "resultados_principais": [
            "Identificação de sítios de ligação específicos nos canais de sódio.",
            "Dependência de dose na inibição da condução nervosa.",
            "Reversibilidade parcial em baixas concentrações."
        ],
        "conclusoes": "Ferramenta valiosa em pesquisa, com riscos à saúde pública.",
        "recomendacoes": [
            "Monitoramento de florações tóxicas.",
            "Biossensores rápidos para detecção em frutos do mar.",
            "Explorar análogos menos tóxicos."
        ],
        "referencias": [
            {"titulo": "Oliveira, F. (2022). Toxinas Marinhas e Saúde Pública.", "link": "https://www.example.com/ref1"},
            {"titulo": "Silva, A.P. et al. (2023). Saxitoxin Review.", "link": "https://www.example.com/ref2"},
        ],
        "report": "reports/saxitoxina.pdf"
    },
    {
        "slug":"tetrodotoxina","titulo":"Tetrodotoxina","categoria":"Toxinas Naturais","origem":"Baiacu e outros","risco":"~1.200× mais tóxica que cianeto","aplicacao":"Bloqueadores de canais iônicos",
        "summary":"Ferramenta em fisiologia neuronal e analgesia experimental.","autor_lider": "Dr. Pedro Costa","data_publicacao": "2023-09-01",
        "metodologia":"Análise in vivo neuro/cárdio","resultados_principais":["Alta afinidade por canais de sódio"],
        "conclusoes":"Potente neurotoxina; potencial uso anestésico","recomendacoes":["Reduzir toxicidade para uso clínico"],
        "referencias":[{"titulo":"Santos, M. (2021). Animais Marinhos.","link":"https://www.example.com/ref3"}],
        "report":"reports/tetrodotoxina.pdf"
    },
    # ... (outros itens mantidos iguais aos seus) ...
]

@app.get("/pesquisas")
def pesquisas_list():
    level = session.get("user_level", 1)
    if level < 2:
        log_event(status="access_denied", user_name=session.get("user_name"), note="Tentativa de acessar pesquisas sem Nível 2+")
        return redirect(url_for("overview"))
    return render_template("pesquisas_list_dark.html", level=level, level_label=LEVEL_LABELS.get(level), docs=RESEARCH)

@app.get("/pesquisas/<slug>")
def pesquisa_detail(slug):
    level = session.get("user_level", 1)
    if level < 2:
        log_event(status="access_denied", user_name=session.get("user_name"), note=f"Tentativa de acessar pesquisa {slug} sem Nível 2+")
        return redirect(url_for("overview"))

    doc = next((d for d in RESEARCH if d["slug"] == slug), None)
    if not doc:
        log_event(status="not_found", user_name=session.get("user_name"), note=f"Pesquisa {slug} não encontrada")
        return redirect(url_for("pesquisas_list"))

    try:
        return render_template(f"reports/{slug}.html", level=level, level_label=LEVEL_LABELS.get(level), doc=doc)
    except TemplateNotFound:
        return render_template("pesquisa_detail_dark.html", level=level, level_label=LEVEL_LABELS.get(level), doc=doc)

# ------------------ ROTAS DE API ------------------
@app.post("/api/verify_enroll")
def api_verify_enroll():
    data = request.get_json(force=True)
    img64 = data.get("image_b64")
    if not img64:
        log_event(status="api_error", note="api_verify_enroll: Imagem não enviada.")
        return jsonify({"ok": False, "error": "Imagem não enviada."}), 400

    img = b64_to_image(img64)
    label, conf, bbox = None, None, None
    try:
        label, conf, bbox = predict_face(img)
    except Exception as e:
        log_event(status="api_error", note=f"api_verify_enroll: Erro ao prever rosto: {e}")
        return jsonify({"ok": False, "error": "Erro no processamento da imagem."}), 500

    if label is None:
        log_event(status="enroll_gate_failed", note="api_verify_enroll: Rosto não detectado")
        return jsonify({"ok": True, "match": False, "reason": "Rosto não detectado", "require_admin": True, "bbox": bbox}), 200

    if conf is not None and float(conf) <= float(LBPH_THRESHOLD):
        lm = load_label_map()
        name = lm.get(label, "Usuário reconhecido")
        users = get_users()
        user = next((u for u in users if u["name"] == name), None)
        level = user["level"] if user else 1
        level_label = LEVEL_LABELS.get(level, f"Nível {level}")
        log_event(status="enroll_gate_face", user_name=name, score=float(conf), note=f"{level_label}")

        if level == 3:
            session["user_name"] = name
            session["user_level"] = level
            return jsonify({"ok": True, "match": True, "is_n3": True, "name": name, "level": level, "bbox": bbox, "redirect": url_for("enroll_form")})
        else:
            return jsonify({"ok": True, "match": True, "is_n3": False, "name": name, "level": level, "bbox": bbox, "require_admin": True}), 200
    else:
        log_event(status="enroll_gate_failed", score=float(conf), note="api_verify_enroll: Confiança acima do threshold")
        return jsonify({"ok": True, "match": False, "reason": "Sem correspondência", "require_admin": True, "bbox": bbox}), 200

@app.post("/admin-login")
def admin_login():
    data = request.get_json(force=True)
    user = data.get("user", "").strip()
    pw = data.get("password", "").strip()

    if user == ADMIN_USER and pw == ADMIN_PASS:
        session["admin_ok"] = True
        log_event(status="admin_login_ok", user_name="admin")
        return jsonify({"ok": True, "redirect": url_for("enroll_form")})

    log_event(status="admin_login_failed", note="admin-login: Credenciais inválidas")
    return jsonify({"ok": False, "error": "Credenciais inválidas"}), 401

@app.post("/api/enroll")
def api_enroll():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    try:
        level = int(data.get("level", 1))
    except ValueError:
        level = 1
        log_event(status="api_error", note=f"api_enroll: Nível inválido; padrão 1. Recebido: {data.get('level')}")

    img64 = data.get("image_b64")
    if not name or not img64:
        log_event(status="api_error", note="api_enroll: Nome ou imagem não fornecidos.")
        return jsonify({"ok": False, "error": "Nome e imagem são obrigatórios."}), 400

    if not (session.get("user_level") == 3 or session.get("admin_ok")):
        log_event(status="access_denied", user_name=session.get("user_name"), note="api_enroll: Tentativa de cadastro sem permissão.")
        return jsonify({"ok": False, "error": "Somente Nível 3 ou Admin podem cadastrar/editar."}), 403

    img = b64_to_image(img64)

    label, conf, bbox = None, None, None
    try:
        label, conf, bbox = predict_face(img)
    except Exception as e:
        log_event(status="api_error", note=f"api_enroll: Erro ao prever rosto durante cadastro: {e}")
        pass

    if label is not None and conf is not None and float(conf) <= float(LBPH_THRESHOLD):
        lm = load_label_map()
        current_name = lm.get(label)
        if current_name:
            updated_level_flag = False
            try:
                updated_level_flag = bool(update_user_level(current_name, level))
            except Exception as e:
                log_event(status="api_error", user_name=current_name, note=f"api_enroll: Erro ao atualizar nível: {e}")

            saved_sample_flag = False
            try:
                path_rel_update = save_face_image(current_name, img, level)
                if path_rel_update:
                    saved_sample_flag = True
                    if callable(update_user_image_path):
                        try:
                            update_user_image_path(current_name, path_rel_update)
                        except Exception as e:
                            log_event(status="api_error", user_name=current_name, note=f"api_enroll: Erro ao atualizar image_path: {e}")
            except Exception as e:
                log_event(status="api_error", user_name=current_name, note=f"api_enroll: Erro ao salvar nova amostra: {e}")

            try:
                train_model()
            except Exception as e:
                log_event(status="api_error", note=f"api_enroll: Erro ao treinar após atualização: {e}")

            log_event(
                status="enroll_update_level",
                user_name=current_name,
                note=f"Nível para {level} e amostra atualizados. Atualizou nível: {updated_level_flag}, Nova amostra: {saved_sample_flag}"
            )
            return jsonify({
                "ok": True,
                "updated_only": True,
                "name": current_name,
                "level": level,
                "saved_sample": saved_sample_flag,
                "note": "Nível de acesso atualizado para rosto já cadastrado."
            })

    # novo cadastro
    path_rel = save_face_image(name, img, level)
    if not path_rel:
        log_event(status="enroll_failed", user_name=name, note="api_enroll: Rosto não detectado na imagem fornecida.")
        return jsonify({"ok": False, "error": "Rosto não detectado na imagem fornecida."}), 200

    try:
        sig = inspect.signature(add_user)
        if len(sig.parameters) == 3:
            add_user(name, level, path_rel)
        else:
            add_user({"name": name, "level": level, "image_path": path_rel})
    except Exception as e:
        log_event(status="enroll_failed_db", user_name=name, note=f"api_enroll: Erro ao adicionar usuário no DB: {e}")
        return jsonify({"ok": False, "error": f"Falha ao salvar usuário no banco: {e}"}), 500

    try:
        train_model()
    except Exception as e:
        log_event(status="api_error", note=f"api_enroll: Erro ao treinar após novo cadastro: {e}")

    log_event(status="enroll_ok", user_name=name, note=f"Novo usuário cadastrado, nível {level}.")
    return jsonify({"ok": True, "note": "Novo usuário cadastrado com sucesso."})

@app.post("/api/verify")
def api_verify():
    data = request.get_json(force=True)
    img64 = data.get("image_b64")
    if not img64:
        log_event(status="api_error", note="api_verify: Imagem não enviada.")
        return jsonify({"ok": False, "error": "Imagem não enviada."}), 400

    img = b64_to_image(img64)
    label, conf, bbox = None, None, None
    try:
        label, conf, bbox = predict_face(img)
    except Exception as e:
        log_event(status="api_error", note=f"api_verify: Erro ao prever rosto: {e}")
        return jsonify({"ok": False, "error": "Erro no processamento da imagem."}), 500

    if label is None:
        log_event(status="auth_failed", note="api_verify: Rosto não detectado ou modelo vazio")
        return jsonify({"ok": True, "match": False, "reason": "Rosto não detectado ou modelo vazio", "bbox": bbox}), 200

    if conf is not None and float(conf) <= float(LBPH_THRESHOLD):
        lm = load_label_map()
        name = lm.get(label, "Usuário reconhecido")
        users = get_users()
        user = next((u for u in users if u["name"] == name), None)
        level = user["level"] if user else 1
        level_label = LEVEL_LABELS.get(level, f"Nível {level}")

        session["user_name"] = name
        session["user_level"] = level

        log_event(status="auth_ok", user_name=name, score=float(conf), note=level_label)
        return jsonify({
            "ok": True,
            "match": True,
            "name": name,
            "level": level,
            "level_label": level_label,
            "distance": float(conf),
            "bbox": bbox,
            "redirect": url_for("overview")
        })
    else:
        log_event(status="auth_failed", score=float(conf), note="api_verify: Confiança acima do threshold")
        return jsonify({"ok": True, "match": False, "reason": "Sem correspondência", "bbox": bbox}), 200

@app.get("/logout")
def logout():
    session.clear()
    log_event(status="logout", note="Usuário deslogado.")
    return redirect(url_for("landing"))

# ---- Main somente para rodar localmente (no Render use Gunicorn) ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False, use_reloader=False)
