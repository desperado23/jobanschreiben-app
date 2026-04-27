import streamlit as st
import requests
import json
import logging
import io

# PDF-Bibliotheken (mit Fallback)
try:
    import pdfplumber
    PDF_LIB = "pdfplumber"
except ImportError:
    try:
        import PyPDF2
        PDF_LIB = "pypdf2"
    except ImportError:
        PDF_LIB = None

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Konfiguration
# ============================================
VENICE_API_URL = "https://api.venice.ai/api/v1/chat/completions"
PRIMARY_MODEL = "llama-3.3-70b"
FALLBACK_MODEL = "mixtral-8x22b"
MAX_PDF_SIZE_MB = 5
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024

# ============================================
# Streamlit Page Config
# ============================================
st.set_page_config(
    page_title="CoverAI - Powered by Venice",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# Custom CSS - Venice Style
# ============================================
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }

    h1 {
        color: #00963F !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    h2, h3 { color: #1a1a1a !important; }

    .stButton > button {
        background-color: #00C853;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #00963F;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 200, 83, 0.3);
    }

    /* Secondary Button (Reset) */
    .stButton > button[kind="secondary"] {
        background-color: #FFFFFF;
        color: #666;
        border: 1px solid #E0E0E0;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #F5F5F5;
        color: #1a1a1a;
        border-color: #00C853;
    }

    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        font-family: 'Inter', sans-serif;
    }

    .stTextArea textarea:focus {
        border-color: #00C853;
        box-shadow: 0 0 0 2px rgba(0, 200, 83, 0.1);
    }

    .stSelectbox > div > div { border-radius: 8px; }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #F4FBF6;
        border: 2px dashed #00C853;
        border-radius: 8px;
        padding: 0.5rem;
    }

    .result-box {
        background-color: #F4FBF6;
        border-left: 4px solid #00C853;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        white-space: pre-wrap;
        font-family: 'Georgia', serif;
        line-height: 1.6;
    }

    .caption-text {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .pdf-success {
        background-color: #E8F5E9;
        color: #00963F;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Session State Initialisierung
# ============================================
if "cv_text_value" not in st.session_state:
    st.session_state.cv_text_value = ""
if "job_text_value" not in st.session_state:
    st.session_state.job_text_value = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = None

# ============================================
# PDF-Extraktion
# ============================================
def extract_text_from_pdf(uploaded_file) -> tuple[bool, str]:
    """
    Extrahiert Text aus einer hochgeladenen PDF-Datei.
    Returns: (success, text_or_error_message)
    """
    if PDF_LIB is None:
        return False, "Keine PDF-Bibliothek installiert. Bitte 'pdfplumber' oder 'PyPDF2' installieren."

    # Dateigröße prüfen
    if uploaded_file.size > MAX_PDF_SIZE_BYTES:
        return False, f"Datei zu groß ({uploaded_file.size / 1024 / 1024:.1f} MB). Maximum: {MAX_PDF_SIZE_MB} MB."

    try:
        file_bytes = uploaded_file.read()
        text_parts = []

        if PDF_LIB == "pdfplumber":
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                if len(pdf.pages) == 0:
                    return False, "PDF enthält keine Seiten."
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

        elif PDF_LIB == "pypdf2":
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            if len(reader.pages) == 0:
                return False, "PDF enthält keine Seiten."
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        full_text = "\n\n".join(text_parts).strip()

        if not full_text:
            return False, "Aus der PDF konnte kein Text extrahiert werden. Möglicherweise handelt es sich um ein Bild-PDF (Scan). Bitte kopiere den Text manuell."

        return True, full_text

    except Exception as e:
        logger.error(f"PDF-Extraktion fehlgeschlagen: {str(e)}")
        return False, f"PDF konnte nicht gelesen werden: {str(e)}"


# ============================================
# API Helper Funktion (mit Cache)
# ============================================
@st.cache_data(show_spinner=False, ttl=3600)
def generate_cover_letter(job_text: str, cv_text: str, tone: str) -> tuple[bool, str]:
    """Generiert ein Anschreiben über die Venice.ai API."""
    api_key = st.secrets.get("VENICE_API_KEY")
    if not api_key:
        return False, "API-Key nicht gefunden. Bitte in st.secrets konfigurieren."

    # NEUER, SCHLAGKRÄFTIGERER SYSTEM-PROMPT
    system_prompt = (
        "Du bist ein erfahrener Headhunter. Schreibe ein knappes, überzeugendes Anschreiben "
        "(max 150 Wörter). Keine Floskeln wie 'ich bewerbe mich um' oder 'ich freue mich auf "
        "ein persönliches Gespräch'. Starte direkt mit der Wertbeitrag-Aussage. Konkrete Bezüge "
        "zur Stellenanzeige. Professionell, aber modern und authentisch."
    )

    user_prompt = f"""STELLENANZEIGE:
{job_text}

MEINE ERFAHRUNGEN:
{cv_text}

GEWÜNSCHTER TONFALL: {tone}

Schreibe jetzt das Anschreiben auf Deutsch. Maximum 150 Wörter. Direkt, präzise, mit konkretem Bezug zur Stelle."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.75,
        "max_tokens": 600
    }

    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        payload["model"] = model
        try:
            response = requests.post(
                VENICE_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return True, content.strip()
            else:
                logger.warning(f"Model {model} failed: {response.status_code} - {response.text}")
                continue

        except requests.exceptions.Timeout:
            logger.error(f"Timeout bei Model {model}")
            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Error bei Model {model}: {str(e)}")
            continue
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Parse Error bei Model {model}: {str(e)}")
            continue

    return False, "Bitte versuche es erneut"


# ============================================
# Reset-Funktion
# ============================================
def reset_app():
    st.session_state.cv_text_value = ""
    st.session_state.job_text_value = ""
    st.session_state.result = None
    st.session_state.pdf_processed = None
    # Cache leeren für neuen Versuch
    generate_cover_letter.clear()


# ============================================
# UI - Header
# ============================================
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 📝")
with col_title:
    st.title("CoverAI")
    st.markdown(
        '<p class="caption-text">Powered by Venice.ai • Dein KI-Anschreiben-Generator</p>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ============================================
# UI - Eingabefelder
# ============================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📋 Stellenanzeige")
    job_text = st.text_area(
        label="Stellenanzeige",
        placeholder="Füge hier den vollständigen Text der Stellenanzeige ein...",
        height=350,
        label_visibility="collapsed",
        key="job_text_value"
    )

with col2:
    st.subheader("👤 Dein Lebenslauf")

    # PDF-Upload
    uploaded_pdf = st.file_uploader(
        f"📎 PDF hochladen (max. {MAX_PDF_SIZE_MB} MB)",
        type=["pdf"],
        key="pdf_uploader",
        help="Lade deinen Lebenslauf als PDF hoch – der Text wird automatisch extrahiert."
    )

    # PDF verarbeiten (nur wenn neu)
    if uploaded_pdf is not None:
        # Datei-Identifier um zu prüfen, ob bereits verarbeitet
        file_id = f"{uploaded_pdf.name}_{uploaded_pdf.size}"

        if st.session_state.pdf_processed != file_id:
            with st.spinner("📄 PDF wird gelesen..."):
                success, extracted = extract_text_from_pdf(uploaded_pdf)

            if success:
                st.session_state.cv_text_value = extracted
                st.session_state.pdf_processed = file_id
                st.markdown(
                    f'<div class="pdf-success">✅ PDF erfolgreich gelesen '
                    f'({len(extracted.split())} Wörter extrahiert)</div>',
                    unsafe_allow_html=True
                )
                st.rerun()
            else:
                st.error(f"❌ {extracted}")

    cv_text = st.text_area(
        label="CV",
        placeholder="Beschreibe deine Erfahrungen, Skills und relevanten Stationen...\n\noder lade oben eine PDF hoch.",
        height=280,
        label_visibility="collapsed",
        key="cv_text_value"
    )

# ============================================
# UI - Tonfall + Buttons
# ============================================
st.markdown("<br>", unsafe_allow_html=True)
col_tone, col_btn_gen, col_btn_reset = st.columns([2, 2, 1])

with col_tone:
    tone = st.selectbox(
        "🎨 Tonfall wählen",
        options=["formell", "modern", "kreativ"],
        index=1,
        help="Wähle den passenden Tonfall für die Branche und Position."
    )

with col_btn_gen:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("✨ Anschreiben generieren", type="primary")

with col_btn_reset:
    st.markdown("<br>", unsafe_allow_html=True)
    reset_btn = st.button("🔄 Reset", type="secondary", on_click=reset_app)

# ============================================
# Generierungs-Logik
# ============================================
if generate_btn:
    if not job_text.strip() or not cv_text.strip():
        st.warning("⚠️ Bitte fülle sowohl die Stellenanzeige als auch deine Erfahrungen aus.")
    else:
        with st.spinner("✍️ Venice schreibt dein Anschreiben... (10-30 Sekunden)"):
            success, result = generate_cover_letter(job_text, cv_text, tone)

        if success:
            st.session_state.result = result
        else:
            st.session_state.result = None
            st.error(f"❌ {result}")

# ============================================
# Ergebnis-Anzeige
# ============================================
if st.session_state.result:
    st.markdown("---")
    st.success("✅ Anschreiben erfolgreich generiert!")
    st.markdown("### 📄 Dein Anschreiben")

    # Schöne Vorschau
    st.markdown(
        f'<div class="result-box">{st.session_state.result}</div>',
        unsafe_allow_html=True
    )

    # Code-Block mit eingebautem Copy-Button von Streamlit
    st.markdown("#### 📋 Zum Kopieren")
    st.caption("Klicke auf das Copy-Icon oben rechts in der Box, um den Text in die Zwischenablage zu kopieren.")
    st.code(st.session_state.result, language=None)

    # Download + Stats
    col_dl, col_stats = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="⬇️ Als TXT herunterladen",
            data=st.session_state.result,
            file_name="anschreiben.txt",
            mime="text/plain"
        )
    with col_stats:
        word_count = len(st.session_state.result.split())
        char_count = len(st.session_state.result)
        st.caption(
            f"📊 **Wörter:** {word_count} • **Zeichen:** {char_count} • "
            f"**Tonfall:** {tone.capitalize()}"
        )

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #888; font-size: 0.85rem;">'
    'Built with ❤️ using Streamlit & Venice.ai • Deine Daten bleiben privat'
    '</p>',
    unsafe_allow_html=True
)
