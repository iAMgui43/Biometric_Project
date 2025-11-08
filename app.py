from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from jinja2 import TemplateNotFound
import base64, io, inspect
from PIL import Image
import numpy as np

# Módulos auxiliares
from db import get_users, add_user, get_logs, log_event, update_user_level  # update_user_level deve existir em db.py
# Tenta importar (opcional) update_user_image_path para atualizar a imagem principal de um usuário
try:
    from db import update_user_image_path  # Pode não existir; tratamos no código
except ImportError: # Use ImportError para importação de módulos
    update_user_image_path = None # Garante que a variável exista mesmo que a importação falhe

from face_utils import save_face_image, predict_face, train_model, LBPH_THRESHOLD, load_label_map

app = Flask(__name__)
app.secret_key = "dev-secret-change-me-stronger-key"  # MUDE ISSO EM PRODUÇÃO PARA UMA CHAVE FORTE
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Ok para desenvolvimento local sem HTTPS

LEVEL_LABELS = {1: "Nível 1 (Geral)", 2: "Nível 2 (Diretoria)", 3: "Nível 3 (Ministro)"}

# Credenciais de Administrador (APENAS PARA DESENVOLVIMENTO - MUDAR EM PRODUÇÃO)
ADMIN_USER = "admin"
ADMIN_PASS = "0000"

def b64_to_image(b64data: str):
    """Converte uma string base64 de imagem (dataURL ou cru) para um array numpy BGR (formato OpenCV)."""
    if "," in b64data:
        b64data = b64data.split(",", 1)[1] # Remove o cabeçalho "data:image/jpeg;base64,"
    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)[:, :, ::-1] # Converte PIL RGB para OpenCV BGR

# ------------------ ROTAS DE PÁGINAS (FRONTEND) ------------------

@app.get("/")
def landing():
    """Página de entrada/landing page."""
    return render_template("landing_dark.html")

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
    """Portão de acesso para o formulário de cadastro/edição de usuários.
    Exige autenticação de Nível 3 ou de Administrador."""
    return render_template("enroll_gate_dark.html", threshold=LBPH_THRESHOLD)

@app.get("/enroll-form")
def enroll_form():
    """Formulário para cadastrar um novo rosto ou atualizar um usuário existente.
    Acesso restrito a usuários de Nível 3 ou Administradores autenticados."""
    if session.get("user_level") == 3 or session.get("admin_ok"):
        return render_template("enroll.html", level_labels=LEVEL_LABELS)
    # Se não tiver permissão, redireciona de volta para o portão de acesso
    return redirect(url_for("enroll_gate"))

@app.get("/overview")
def overview():
    """Dashboard de visão geral. Primeira tela após a validação bem-sucedida."""
    name = session.get("user_name")
    level = session.get("user_level", 1)
    # Garante que o usuário esteja logado para acessar
    if not name:
        log_event(status="access_denied", note="Tentativa de acessar overview sem sessão")
        return redirect(url_for("landing"))
    return render_template("overview_dark.html", name=name, level=level, level_label=LEVEL_LABELS.get(level))

@app.get("/home")
def home_view():
    """Página inicial mais personalizada para o usuário logado."""
    name = session.get("user_name")
    level = session.get("user_level", 1)
    # Garante que o usuário esteja logado para acessar
    if not name:
        log_event(status="access_denied", note="Tentativa de acessar home sem sessão")
        return redirect(url_for("landing"))
    return render_template("home_dark.html", name=name, level=level, level_label=LEVEL_LABELS.get(level))

# ------------------ DADOS DE PESQUISA ------------------

# Dados de pesquisa, como no relatório detalhado. Inclui os novos campos.
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
        "metodologia": "Estudos in vitro com neurônios cultivados e ensaios in vivo em modelos murinos para avaliar toxicidade e mecanismo de ação. Análise cromatográfica para purificação da toxina.",
        "resultados_principais": [
            "Identificação de sítios de ligação específicos nos canais de sódio.",
            "Demonstração da dependência de dose na inibição da condução nervosa.",
            "Confirmação da reversibilidade parcial dos efeitos tóxicos em baixas concentrações."
        ],
        "conclusoes": "A saxitoxina é um potente bloqueador de canais de sódio, com implicações significativas na neurofisiologia e na segurança alimentar. Sua estrutura permite alta especificidade, tornando-a uma ferramenta valiosa em pesquisa, mas com riscos de saúde pública consideráveis.",
        "recomendacoes": [
            "Intensificar o monitoramento de florações de algas tóxicas em áreas costeiras.",
            "Desenvolver biossensores mais rápidos para detecção da toxina em frutos do mar.",
            "Explorar análogos da saxitoxina com menor toxicidade para potenciais aplicações terapêuticas."
        ],
        "referencias": [
            {"titulo": "Oliveira, F. (2022). Toxinas Marinhas e Saúde Pública. Editora Científica.", "link": "https://www.example.com/ref1"},
            {"titulo": "Silva, A.P. et al. (2023). Saxitoxin: A Review of its Mechanisms and Therapeutic Potential. Journal of Toxicology.", "link": "https://www.example.com/ref2"}
        ],
        "report": "reports/saxitoxina.pdf" # Este arquivo PDF deve existir em static/reports/
    },
    # Adicione outros itens de pesquisa aqui, seguindo a nova estrutura para um relatório completo
    # Se um campo novo não for preenchido, o template usará o valor 'default'
    {
        "slug":"tetrodotoxina","titulo":"Tetrodotoxina","categoria":"Toxinas Naturais","origem":"Baiacu e outros animais marinhos","risco":"~1.200× mais tóxica que o cianeto; risco de parada cardiorrespiratória","aplicacao":"Pesquisa em anestésicos e bloqueadores de canais iônicos","summary":"Altíssima potência; usada como ferramenta em fisiologia neuronal e analgesia experimental.","autor_lider": "Dr. Pedro Costa", "data_publicacao": "2023-09-01", "metodologia": "Análise in vivo de efeitos neurotóxicos e cardíacos.", "resultados_principais": ["Alta afinidade por canais de sódio dependentes de voltagem."], "conclusoes": "A tetrodotoxina é uma neurotoxina poderosa, com potencial uso em anestesia local.", "recomendacoes": ["Estudar a modificação molecular para reduzir a toxicidade e ampliar o uso clínico."], "referencias": [{"titulo": "Santos, M. (2021). Toxicologia de Animais Marinhos. Editora Oceano.", "link": "https://www.example.com/ref3"}], "report":"reports/tetrodotoxina.pdf"
    },
    {"slug":"ricina","titulo":"Ricina","categoria":"Toxinas Naturais","origem":"Sementes de mamona (Ricinus communis)","risco":"Inibe ribossomos; dose letal ~0,2 mg/kg; potencial uso como agente biológico","aplicacao":"Imunotoxinas para oncologia (pesquisa)","summary":"Toxina proteica que bloqueia síntese proteica; investigada como componente terapêutico experimental.","autor_lider": "Profa. Carla Mendes", "data_publicacao": "2023-07-20", "metodologia": "Estudos de inibição de síntese proteica em cultura de células.", "resultados_principais": ["Inibição eficaz da síntese proteica em células cancerígenas."], "conclusoes": "A ricina demonstra potencial como componente em imunotoxinas para tratamento de câncer.", "recomendacoes": ["Otimizar a entrega direcionada da toxina para minimizar efeitos adversos sistêmicos."], "referencias": [{"titulo": "Mendes, C. et al. (2023). Ricin-Based Immunotoxins in Cancer Therapy. J. Biomedical Research.", "link": "https://www.example.com/ref4"}], "report":"reports/ricina.pdf"
    },
    {"slug":"dioxinas","titulo":"Dioxinas","categoria":"Agentes Químicos Industriais","origem":"Subprodutos de processos industriais e incineração","risco":"Altamente carcinogênicas e teratogênicas; bioacumulativas","aplicacao":"Sem uso benéfico; foco em contenção e monitoramento","summary":"Contaminantes persistentes de alto impacto ambiental e à saúde; gestão ambiental rigorosa.","autor_lider": "Equipe de Monitoramento Ambiental", "data_publicacao": "2023-06-01", "metodologia": "Análise de amostras ambientais e tecidos biológicos.", "resultados_principais": ["Níveis elevados de dioxinas em áreas próximas a incineradores."], "conclusoes": "A contaminação por dioxinas persiste em ecossistemas urbanos e rurais.", "recomendacoes": ["Implementar tecnologias de incineração mais limpas e monitoramento contínuo."], "referencias": [{"titulo": "Agência Ambiental. (2023). Relatório Anual de Dioxinas no Ambiente. Relatório Técnico.", "link": "https://www.example.com/ref5"}], "report":"reports/dioxinas.pdf"
    },
    {"slug":"mercurio-metilado","titulo":"Mercúrio Metilado","categoria":"Agentes Químicos Industriais","origem":"Metilação de mercúrio inorgânico em ecossistemas aquáticos","risco":"Neurotoxina persistente; bioacumula em peixes e frutos do mar","aplicacao":"Uso extremamente controlado em pesquisa de catalisadores","summary":"Exige políticas de controle e vigilância por impactos neurológicos e bioacumulação.","autor_lider": "Dr. Fernando Rocha", "data_publicacao": "2023-05-10", "metodologia": "Estudo de bioacumulação em cadeias alimentares aquáticas.", "resultados_principais": ["Concentrações elevadas de mercúrio metilado em peixes predadores."], "conclusoes": "O mercúrio metilado representa um risco significativo para a saúde humana através do consumo de peixe.", "recomendacoes": ["Educar a população sobre o consumo seguro de peixes e controlar as emissões de mercúrio industrial."], "referencias": [{"titulo": "Rocha, F. et al. (2023). Mercury Methylation and Bioaccumulation in Aquatic Systems. Environmental Science Journal.", "link": "https://www.example.com/ref6"}], "report":"reports/mercurio-metilado.pdf"
    },
    {"slug":"nicotinoides-sinteticos","titulo":"Nicotinoides Sintéticos","categoria":"Agentes Químicos Industriais","origem":"Inseticidas agrícolas","risco":"Associados ao colapso de polinizadores; impacto ecossistêmico","aplicacao":"Proteção de cultivos em contextos controlados","summary":"Risco a abelhas e biodiversidade; regulamentação e boas práticas são essenciais.","autor_lider": "Dra. Sofia Lima", "data_publicacao": "2023-04-25", "metodologia": "Estudos de campo sobre o impacto em populações de abelhas.", "resultados_principais": ["Redução na densidade de colmeias em áreas tratadas com neonicotinoides."], "conclusoes": "Os neonicotinoides representam uma ameaça à saúde dos polinizadores.", "recomendacoes": ["Promover o uso de alternativas e restrições no uso de neonicotinoides, especialmente durante a floração."], "referencias": [{"titulo": "Lima, S. et al. (2023). Neonicotinoids and Bee Decline: A Global Perspective. Ecological Entomology.", "link": "https://www.example.com/ref7"}], "report":"reports/nicotinoides-sinteticos.pdf"
    },
    {"slug":"h5n1-modificado","titulo":"H5N1 Modificado","categoria":"Patógenos Modificados","origem":"Variante de influenza aviária alterada em laboratório","risco":"Potencial pandêmico com alta letalidade; aumento de transmissibilidade","aplicacao":"Desenvolvimento de vacinas e estudos imunológicos","summary":"Pesquisa de alto controle para vigilância e preparação a pandemias.","autor_lider": "Dr. Ricardo Nunes", "data_publicacao": "2023-03-15", "metodologia": "Ensaios de ganho de função em modelos animais.", "resultados_principais": ["Demonstração de maior transmissibilidade por aerossol em furões."], "conclusoes": "A manipulação genética do H5N1 pode criar cepas com potencial pandêmico significativo.", "recomendacoes": ["Reforçar as diretrizes de biossegurança para pesquisas de ganho de função e a vigilância global."], "referencias": [{"titulo": "Nunes, R. et al. (2023). Enhanced Pathogenicity of H5N1 Influenza Variants. Virology Journal.", "link": "https://www.example.com/ref8"}], "report":"reports/h5n1-modificado.pdf"
    },
    {"slug":"bacillus-anthracis-resistente","titulo":"Bacillus anthracis Resistente","categoria":"Patógenos Modificados","origem":"Cepa de antraz com resistência múltipla","risco":"Elevado risco de uso indevido/bioterrorismo","aplicacao":"P&D de antídotos e vacinas","summary":"Necessita biossegurança máxima e rastreabilidade.","autor_lider": "Dra. Laura Gomes", "data_publicacao": "2023-02-01", "metodologia": "Identificação e caracterização de genes de resistência a antibióticos.", "resultados_principais": ["Cepa com resistência a múltiplos antibióticos comuns."], "conclusoes": "A emergência de cepas de B. anthracis resistentes a antibióticos representa uma séria ameaça de bioterrorismo.", "recomendacoes": ["Desenvolvimento urgente de novas terapias e vacinas de segunda geração, e fortalecimento da capacidade de resposta a emergências biológicas."], "referencias": [{"titulo": "Gomes, L. et al. (2023). Multi-Drug Resistant Bacillus anthracis: A Threat Assessment. Journal of Infectious Diseases.", "link": "https://www.example.com/ref9"}], "report":"reports/bacillus-anthracis-resistente.pdf"
    },
    {"slug":"candida-auris-termotolerante","titulo":"Candida auris Termotolerante","categoria":"Patógenos Modificados","origem":"Fungo hospitalar emergente tolerante a altas temperaturas","risco":"Surtos hospitalares; resistência a antifúngicos","aplicacao":"Pesquisa de mecanismos de resistência e novas terapias","summary":"Patógeno desafiador para controle infeccioso; exige vigilância.","autor_lider": "Dr. Carlos Eduardo", "data_publicacao": "2023-01-10", "metodologia": "Testes de suscetibilidade a antifúngicos e estudos de termotolerância.", "resultados_principais": ["Capacidade de crescimento em temperaturas elevadas e resistência a classes de antifúngicos."], "conclusoes": "C. auris representa uma ameaça crescente à saúde global devido à sua resistência e persistência.", "recomendacoes": ["Aprimorar as práticas de controle de infecção hospitalar e acelerar a pesquisa de novos agentes antifúngicos."], "referencias": [{"titulo": "Eduardo, C. et al. (2023). Thermal Adaptation and Antifungal Resistance in Candida auris. Mycology Research.", "link": "https://www.example.com/ref10"}], "report":"reports/candida-auris-termotolerante.pdf"
    },
    {"slug":"biologia-sintetica","titulo":"Biologia Sintética","categoria":"Tema Sensível","origem":"Engenharia genética avançada","risco":"Criação de organismos patogênicos/artificiais; riscos de dual use","aplicacao":"Terapias gênicas, bioengenharia segura e bioremediação","summary":"Potencial transformador com riscos regulatórios e éticos.","autor_lider": "Comitê de Bioética e Inovação", "data_publicacao": "2022-12-05", "metodologia": "Análise de riscos e benefícios de tecnologias emergentes em biologia sintética.", "resultados_principais": ["Identificação de lacunas regulatórias e dilemas éticos na criação de novas formas de vida."], "conclusoes": "A biologia sintética oferece grandes promessas, mas exige uma estrutura regulatória e ética robusta para mitigar riscos.", "recomendacoes": ["Desenvolver diretrizes globais para pesquisa de biologia sintética e fomentar o diálogo público sobre suas implicações."], "referencias": [{"titulo": "Comitê de Bioética. (2022). O Futuro da Biologia Sintética: Riscos e Oportunidades. Relatório de Política.", "link": "https://www.example.com/ref11"}], "report":"reports/biologia-sintetica.pdf"
    }
]

@app.get("/pesquisas")
def pesquisas_list():
    """Lista todas as pesquisas, acessível apenas para níveis >= 2."""
    level = session.get("user_level", 1)
    if level < 2:
        log_event(status="access_denied", user_name=session.get("user_name"), note="Tentativa de acessar pesquisas sem Nível 2+")
        return redirect(url_for("overview"))
    return render_template("pesquisas_list_dark.html", level=level, level_label=LEVEL_LABELS.get(level), docs=RESEARCH)

@app.get("/pesquisas/<slug>")
def pesquisa_detail(slug):
    """Exibe os detalhes de uma pesquisa específica. Acessível apenas para níveis >= 2."""
    level = session.get("user_level", 1)
    if level < 2:
        log_event(status="access_denied", user_name=session.get("user_name"), note=f"Tentativa de acessar pesquisa {slug} sem Nível 2+")
        return redirect(url_for("overview"))
    
    doc = next((d for d in RESEARCH if d["slug"] == slug), None)
    if not doc:
        log_event(status="not_found", user_name=session.get("user_name"), note=f"Pesquisa {slug} não encontrada")
        return redirect(url_for("pesquisas_list")) # Redireciona se o documento não for encontrado
    
    try:
        # Tenta renderizar um template específico para o slug (ex: reports/saxitoxina.html)
        return render_template(f"reports/{slug}.html", level=level, level_label=LEVEL_LABELS.get(level), doc=doc)
    except TemplateNotFound:
        # Se o template específico não existir, usa o template genérico detalhado
        return render_template("pesquisa_detail_dark.html", level=level, level_label=LEVEL_LABELS.get(level), doc=doc)

# ------------------ ROTAS DE API ------------------

@app.post("/api/verify_enroll")
def api_verify_enroll():
    """API para verificar um rosto no 'enroll-gate'. Retorna se o rosto é N3 ou precisa de admin."""
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
        level = user["level"] if user else 1 # Padrão para Nível 1 se o usuário não for encontrado no DB
        level_label = LEVEL_LABELS.get(level, f"Nível {level}")
        log_event(status="enroll_gate_face", user_name=name, score=float(conf), note=f"{level_label}")

        if level == 3:
            # Usuário Nível 3 autenticado no gate, concede acesso ao formulário
            session["user_name"] = name
            session["user_level"] = level
            return jsonify({"ok": True, "match": True, "is_n3": True, "name": name, "level": level, "bbox": bbox, "redirect": url_for("enroll_form")})
        else:
            # Outros níveis são reconhecidos, mas ainda precisam de senha de admin
            return jsonify({"ok": True, "match": True, "is_n3": False, "name": name, "level": level, "bbox": bbox, "require_admin": True}), 200
    else:
        log_event(status="enroll_gate_failed", score=float(conf), note="api_verify_enroll: Confiança acima do threshold")
        return jsonify({"ok": True, "match": False, "reason": "Sem correspondência", "require_admin": True, "bbox": bbox}), 200

@app.post("/admin-login")
def admin_login():
    """API para login do administrador no 'enroll-gate'."""
    data = request.get_json(force=True)
    user = data.get("user", "").strip()
    pw = data.get("password", "").strip()

    if user == ADMIN_USER and pw == ADMIN_PASS:
        session["admin_ok"] = True # Define a flag de admin na sessão
        log_event(status="admin_login_ok", user_name="admin")
        return jsonify({"ok": True, "redirect": url_for("enroll_form")})
    
    log_event(status="admin_login_failed", note="admin-login: Credenciais inválidas")
    return jsonify({"ok": False, "error": "Credenciais inválidas"}), 401 # 401 Unauthorized

@app.post("/api/enroll")
def api_enroll():
    """API para cadastrar um novo usuário ou atualizar o nível/amostra de um rosto existente.
    Acesso restrito."""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    try:
        level = int(data.get("level", 1))
    except ValueError: # Captura erro se 'level' não for um inteiro válido
        level = 1
        log_event(status="api_error", note=f"api_enroll: Nível inválido, padrão para 1. Recebido: {data.get('level')}")

    img64 = data.get("image_b64")

    if not name or not img64:
        log_event(status="api_error", note="api_enroll: Nome ou imagem não fornecidos.")
        return jsonify({"ok": False, "error": "Nome e imagem são obrigatórios."}), 400
    
    # Validação de permissão: Nível 3 ou Admin para cadastrar/editar
    if not (session.get("user_level") == 3 or session.get("admin_ok")):
        log_event(status="access_denied", user_name=session.get("user_name"), note="api_enroll: Tentativa de cadastro sem permissão.")
        return jsonify({"ok": False, "error": "Somente Nível 3 ou Admin podem cadastrar/editar."}), 403 # 403 Forbidden

    img = b64_to_image(img64)

    # 1) Tenta prever se o rosto já existe no modelo para atualização
    label, conf, bbox = None, None, None
    try:
        label, conf, bbox = predict_face(img)
    except Exception as e:
        log_event(status="api_error", note=f"api_enroll: Erro ao prever rosto durante cadastro: {e}")
        # Continua o fluxo como se não tivesse encontrado, pois pode ser um novo rosto
        pass 

    if label is not None and conf is not None and float(conf) <= float(LBPH_THRESHOLD):
        lm = load_label_map()
        current_name = lm.get(label)

        if current_name:
            # Rosto já existe: Atualiza nível e opcionalmente a imagem principal
            updated_level_flag = False
            try:
                # Assumindo que update_user_level retorna True/False ou similar
                updated_level_flag = bool(update_user_level(current_name, level))
            except Exception as e:
                log_event(status="api_error", user_name=current_name, note=f"api_enroll: Erro ao atualizar nível do usuário: {e}")
                
            saved_sample_flag = False
            try:
                # Salva uma nova amostra da face e tenta atualizar o image_path principal
                path_rel_update = save_face_image(current_name, img, level)
                if path_rel_update:
                    saved_sample_flag = True
                    if callable(update_user_image_path): # Verifica se a função existe e é chamável
                        try:
                            update_user_image_path(current_name, path_rel_update)
                        except Exception as e:
                            log_event(status="api_error", user_name=current_name, note=f"api_enroll: Erro ao atualizar path da imagem do usuário: {e}")
            except Exception as e:
                log_event(status="api_error", user_name=current_name, note=f"api_enroll: Erro ao salvar nova amostra de imagem: {e}")

            try:
                train_model() # Retreina o modelo após qualquer alteração
            except Exception as e:
                log_event(status="api_error", note=f"api_enroll: Erro ao treinar modelo após atualização: {e}")

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

    # 2) Se o rosto não foi reconhecido ou não existe no modelo: procede com o novo cadastro
    path_rel = save_face_image(name, img, level) # Salva a imagem na pasta 'faces/'
    if not path_rel:
        log_event(status="enroll_failed", user_name=name, note="api_enroll: Rosto não detectado na imagem fornecida para novo cadastro.")
        return jsonify({"ok": False, "error": "Rosto não detectado na imagem fornecida."}), 200

    # Adiciona o usuário ao banco de dados, com tratamento para diferentes assinaturas da função add_user
    try:
        sig = inspect.signature(add_user)
        if len(sig.parameters) == 3: # Assinatura antiga: add_user(name, level, path)
            add_user(name, level, path_rel)
        else: # Assinatura moderna: add_user({"name": name, "level": level, "image_path": path})
            add_user({"name": name, "level": level, "image_path": path_rel})
    except Exception as e:
        log_event(status="enroll_failed_db", user_name=name, note=f"api_enroll: Erro ao adicionar usuário no DB: {e}")
        return jsonify({"ok": False, "error": f"Falha ao salvar usuário no banco de dados: {e}"}), 500

    try:
        train_model() # Treina o modelo após adicionar um novo rosto
    except Exception as e:
        log_event(status="api_error", note=f"api_enroll: Erro ao treinar modelo após novo cadastro: {e}")

    log_event(status="enroll_ok", user_name=name, note=f"Novo usuário cadastrado, nível {level}.")
    return jsonify({"ok": True, "note": "Novo usuário cadastrado com sucesso."})

@app.post("/api/verify")
def api_verify():
    """API para autenticação facial (login)."""
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
        # Autenticação bem-sucedida
        lm = load_label_map()
        name = lm.get(label, "Usuário reconhecido")
        users = get_users()
        user = next((u for u in users if u["name"] == name), None)
        level = user["level"] if user else 1 # Padrão para Nível 1 se o usuário não for encontrado no DB
        level_label = LEVEL_LABELS.get(level, f"Nível {level}")
        
        # Define os dados do usuário na sessão
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
            "redirect": url_for("overview") # Redireciona para o overview após login
        })
    else:
        # Autenticação falhou: confiança acima do threshold
        log_event(status="auth_failed", score=float(conf), note="api_verify: Confiança acima do threshold")
        return jsonify({"ok": True, "match": False, "reason": "Sem correspondência", "bbox": bbox}), 200

@app.get("/logout")
def logout():
    """Rota para fazer logout do usuário, limpando a sessão."""
    session.clear() # Limpa todos os dados da sessão
    log_event(status="logout", note="Usuário deslogado.")
    return redirect(url_for("landing"))

if __name__ == "__main__":
    # Configurações para rodar o servidor Flask
    # Em desenvolvimento, use 127.0.0.1 para a câmera funcionar sem HTTPS.
    # Para acesso via rede (ex.: 192.168.x.x) e câmera em dispositivos móveis,
    # você PRECISA de HTTPS. Gere certificados com openssl e use ssl_context.
    # Exemplo:
    # openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes
    # app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, ssl_context=("cert.pem","key.pem"))
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)