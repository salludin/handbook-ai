"""
Backend chatbot Handbook (FastAPI + OpenAI)

Install:  pip install fastapi uvicorn openai pymupdf pydantic python-dotenv
Konfigurasi: Isi OPENAI_API_KEY dan APP_TOKEN di file .env
Jalankan: uvicorn handbook:app --host 0.0.0.0 --port 8000

CATATAN: session disimpan di memori, jadi jalankan dengan 1 worker saja.
Jika perlu >1 worker/server, pindahkan SESSIONS ke Redis dengan TTL.
"""

import asyncio
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from dotenv import load_dotenv
from extract_handbook import load_pages
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Muat variabel dari file .env di folder project
load_dotenv()

# ====== KONFIGURASI ======
# Handbook General (Opsional):
# Jika tidak ingin menyertakan handbook umum, cukup beri komentar (#) atau set ke None:
#GENERAL_PDF_PATH = "handbook.pdf"  # Contoh menonaktifkan: GENERAL_PDF_PATH = None atau beri tanda '#' di awal

DIVISION_FILES = {
    "dc": "handbook_dc.pdf",
    "preschool": "handbook_preschool.pdf",
    "primary": "handbook_primary.pdf",
    "secondary": "handbook_secondary.pdf",
}
DIVISION_NAMES = {
    "general": "General / Seluruh Divisi",
    "dc": "Daycare / Development Center (DC)",
    "preschool": "Preschool / TK",
    "primary": "Primary / SD",
    "secondary": "Secondary / SMP-SMA",
}
MODEL = "gpt-4.1-nano"
CONTACT = "School Administration Office / Kantor Sekolah, [isi nomor/email di sini]"
SESSION_TTL_SECONDS = 30 * 60  # sesi idle 30 menit otomatis dihapus
MAX_HISTORY_MESSAGES = 12  # 6 tanya-jawab terakhir yang dikirim ke model
MAX_SESSIONS = 1000
APP_TOKEN = os.getenv("APP_TOKEN", "")
if not APP_TOKEN:
    raise RuntimeError("APP_TOKEN belum diisi di file .env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY belum diisi di file .env")
# =========================

NOT_FOUND_MSG = f"This information could not be found in the handbook. Please contact {CONTACT}."
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ---------- Handbook Multi-Divisi Loader ----------
PAGES_CACHE: dict[str, dict[int, str]] = {}
PAGES_MTIME: dict[str, float] = {}


def get_handbook_pages(filepath: str) -> dict[int, str]:
    """Membaca halaman PDF secara otomatis dan me-reload jika file diunggah/diubah di disk."""
    if not os.path.exists(filepath):
        return {}
    mtime = os.path.getmtime(filepath)
    if filepath not in PAGES_CACHE or PAGES_MTIME.get(filepath) != mtime:
        PAGES_CACHE[filepath] = load_pages(filepath)
        PAGES_MTIME[filepath] = mtime
    return PAGES_CACHE[filepath]


def parse_divisions(val: list[str] | str | None) -> list[str]:
    """Mengurai input divisi menjadi list nama divisi yang valid."""
    all_keys = list(DIVISION_FILES.keys())  # ["dc", "preschool", "primary", "secondary"]
    if not val:
        return ["general"]
    if isinstance(val, str):
        parts = [p.strip().lower() for p in val.split(",") if p.strip()]
        if any(p in ("all", "*", "semua") for p in parts):
            return all_keys
        res = [p for p in parts if p in DIVISION_NAMES]
        return res if res else ["general"]
    if isinstance(val, list):
        parts = [str(p).strip().lower() for p in val if str(p).strip()]
        if any(p in ("all", "*", "semua") for p in parts):
            return all_keys
        res = [p for p in parts if p in DIVISION_NAMES]
        return res if res else ["general"]
    return ["general"]


def get_active_sources(divisions: list[str]) -> dict[str, dict[int, str]]:
    """Mengumpulkan kamus halaman handbook yang aktif untuk sesi ini.
    Jika sesi memilih general dan file general tidak ada/dinonaktifkan, otomatis memuat semua handbook divisi yang tersedia.
    """
    sources: dict[str, dict[int, str]] = {}
    general_path = globals().get("GENERAL_PDF_PATH", None)
    has_general_file = bool(general_path and os.path.exists(general_path))

    effective_divs = list(divisions) if divisions else ["general"]
    if ("general" in effective_divs and not has_general_file) or any(d in ("all", "*", "semua") for d in effective_divs):
        for k in DIVISION_FILES:
            if k not in effective_divs:
                effective_divs.append(k)

    for d in effective_divs:
        d_key = d.lower()
        if d_key != "general" and d_key in DIVISION_FILES:
            d_file = DIVISION_FILES[d_key]
            if d_file and os.path.exists(d_file):
                d_title = "DC" if d_key == "dc" else d_key.title()
                d_pages = get_handbook_pages(d_file)
                if d_pages:
                    sources[d_title] = d_pages

    if has_general_file:
        gen_pages = get_handbook_pages(general_path)
        if gen_pages:
            sources["General"] = gen_pages

    # Fallback darurat: selalu sediakan setidaknya 1 sumber jika ada file handbook di disk
    if not sources:
        for d_key, d_file in DIVISION_FILES.items():
            if d_file and os.path.exists(d_file):
                d_title = "DC" if d_key == "dc" else d_key.title()
                d_pages = get_handbook_pages(d_file)
                if d_pages:
                    sources[d_title] = d_pages

    return sources


def build_system_prompt(divisions: list[str]) -> str:
    divs = divisions if divisions else ["general"]
    div_labels = [DIVISION_NAMES.get(d, d.title()) for d in divs]
    div_context_str = ", ".join(div_labels)

    active_sources = get_active_sources(divs)
    sections = []
    for s_name, pages in active_sources.items():
        text = "\n\n".join(
            f"[{s_name} Handbook - Page {n}]\n{t.strip()}\n[/{s_name} Handbook - Page {n}]"
            for n, t in pages.items()
        )
        sections.append(f"=== {s_name.upper()} HANDBOOK ===\n{text}")

    all_handbook_text = "\n\n".join(sections) if sections else "No specific handbook document is currently active."

    return f"""You are the official Parent & Student Handbook assistant for Mutiara Harapan Islamic School (MHIS).
You are not a cold, mechanical robot. You communicate as a warm, caring, respectful, and wise school partner who truly cares about the well-being of the children and their families, embodying Islamic values, Akhlaqul Karimah, and heartfelt empathy.

CURRENT SESSION CONTEXT:
- The parent has child(ren) enrolled in the following division(s): {div_context_str}.
- Multi-child / Multi-division handling:
  * When a parent asks about a specific child or mentions a division (e.g., "anak saya yang di TK" or "untuk yang di SD"), answer specifically based on that division's handbook and guidelines.
  * If the question is general or applies across their children, guide them comprehensively and clarify any differences between their enrolled divisions (e.g. schedules, homeroom teachers, uniforms, arrival procedures).

PARENT PARTNERSHIP & EMPATHETIC TONE PRINCIPLES:
1. ADDRESS PARENTS WARMLY & RESPECTFULLY:
   - In Indonesian, always address parents affectionately and respectfully as "Ayah/Bunda" (as consistently used in the MHIS handbook).
   - In English, address them warmly as "Dear Ayah/Bunda" or "Dear Parents".
2. EMPATHY FIRST (NOT LIKE A ROBOT):
   - When a parent expresses distress, worry, anger, or reports a sensitive issue (e.g., bullying, conflict, ananda merasa sedih/takut, cedera, kesulitan belajar):
     * NEVER jump directly into cold bureaucratic regulations or dry lists.
     * FIRST, validate their feelings with sincere compassion, reassurance, and empathy (e.g., "Kami sangat memahami kekhawatiran dan rasa cemas Ayah/Bunda. Kenyamanan, rasa aman, dan keselamatan ananda adalah prioritas paling utama bagi kami di MHIS.").
     * Remind them of the core school philosophy: "Parents and MHIS are partners raising the same child" (Orang tua dan sekolah adalah mitra bersama dalam mendidik anak). Ayah/Bunda are not alone, and the school is fully committed to helping resolve the situation safely and constructively.
3. CLEAR, REASSURING & NATURAL GUIDANCE:
   - Guide them through the proper channels and escalation flow naturally, encouragingly, and clearly.
   - For sensitive concerns like bullying or safeguarding:
     * Explain that this is treated with high priority as a sensitive issue.
     * Advise contacting the Homeroom Teacher (Wali Kelas) or reaching out to the Vice Principal Operations and Students' Well-being (VPOSW) / Principal.
     * Recommend communicating via Formal Email (Surel Resmi) or arranging an Appointment (Pertemuan tatap muka pribadi) so the issue can be heard thoroughly and documented with an MoM (Minutes of Meeting).
     * Gently advise protecting ananda by keeping the matter private and avoiding airing it on public social media or parent group chats.
4. ISLAMIC ADAB & COURTESY:
   - Greet warmly with Islamic courtesy (e.g., reply to "Assalamu'alaikum" with "Wa'alaikumussalam wa Rahmatullahi wa Barakatuh").
   - Use polite, constructive, and soothing language (Akhlaqul Karimah) throughout the conversation.

DOCUMENT & CITATION RULES:
5. LANGUAGE: Always respond in the EXACT language used by the user.
6. CONTEXTUAL & SEMANTIC MAPPING: Intelligently understand user queries and map everyday words (bullying, perundungan, diejek, berantem, jam masuk, jam pulang, hari rabu, seragam, dll.) to handbook sections.
7. STRICT MANDATORY CITATIONS (VERY IMPORTANT):
   - Whenever `found=true`, the `citations` list MUST NOT BE EMPTY! You MUST provide at least one citation.
   - Every factual answer, schedule, time, policy, or procedure MUST have a corresponding citation in `citations`:
     * `source`: The exact handbook name (e.g., "Primary", "Preschool", "DC", "Secondary", or "General").
     * `page`: The exact integer page number from the `[<Source> Handbook - Page X]` header (e.g., 13, 14, 28).
     * `quote`: An exact verbatim sentence, clause, or table row from that page confirming the answer (e.g., "07.00 - 07.10 Opening", "13.35 Dismissal", or schedule row). Copy directly from the handbook text provided.
   - If `found=false`, `citations` MUST be empty `[]`.
8. OVERVIEW QUESTIONS: If asked general overview questions (e.g., "bisa bantu apa saja?"), summarize the main handbook sections for the relevant divisions warmly and cite the relevant introductory or table of contents pages.
9. TRUE NOT FOUND: Only set `found=false` if the question is completely outside school matters. Even then, reply politely with warmth and provide contact: {CONTACT}.

HANDBOOK REFERENCES:
{all_handbook_text}
"""


def normalize(s: str) -> str:
    s = s.replace("\u200b", " ").replace("\u00ad", " ").replace("-\n", "")
    s = s.replace("—", " ").replace("–", " ").replace("-", " ")
    s = s.replace("<br>", " ").replace("|", " ")  # pemisah sel/baris tabel markdown
    s = re.sub(r"[^\w\s]", " ", s)  # abaikan tanda baca seperti titik, koma, kurung
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_quote_in_pages(quote: str, page: int, pages: dict[int, str]) -> bool:
    if page not in pages:
        return False
    page_norm = normalize(pages[page])
    q_norm = normalize(quote)
    if not q_norm or len(q_norm) < 3:
        return False
    # 1. Kecocokan langsung
    if q_norm in page_norm:
        return True
    # 2. Kecocokan per segmen (jika kutipan dipisah baris/elipsis)
    segments = [s for s in re.split(r"\n|\.\.\.|…", quote) if s.strip()]
    if segments and any(normalize(seg) in page_norm for seg in segments if len(normalize(seg)) >= 6):
        return True
    # 3. Kecocokan kata kunci (word overlap >= 60%)
    words = [w for w in q_norm.split() if len(w) > 3]
    if words and (sum(1 for w in words if w in page_norm) / len(words) >= 0.6):
        return True
    return False


def resolve_citation(quote: str, page: int, source: str, divisions: list[str]) -> tuple[str, int, str]:
    sources = get_active_sources(divisions)

    # 1. Cek di target_source dan target_page
    target_src = None
    for s_name in sources:
        if source and s_name.lower() in source.lower():
            target_src = s_name
            break

    if target_src and target_src in sources:
        if verify_quote_in_pages(quote, page, sources[target_src]):
            return (target_src, page, quote)
        for p in sources[target_src]:
            if p != page and verify_quote_in_pages(quote, p, sources[target_src]):
                return (target_src, p, quote)

    # 2. Cek di seluruh source lainnya
    for s_name, s_pages in sources.items():
        if verify_quote_in_pages(quote, page, s_pages):
            return (s_name, page, quote)
        for p in s_pages:
            if verify_quote_in_pages(quote, p, s_pages):
                return (s_name, p, quote)

    fallback_src = target_src if target_src else (next(iter(sources.keys())) if sources else "Handbook")
    return (fallback_src, page, quote)


def find_fallback_citations(
    answer: str, question: str, sources: dict[str, dict[int, str]]
) -> list["CitationOut"]:
    """Jika model mengembalikan found=True tetapi citations kosong, cari halaman & kutipan paling relevan dari handbook yang aktif."""
    if not sources:
        return []

    combined = f"{question} {answer}"
    time_matches = re.findall(r"\b\d{1,2}[.:]\d{2}\b", combined)
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", combined)]
    stopwords = {
        "pada", "hari", "siswa", "masuk", "pulang", "sesuai", "jadwal", "telah",
        "ditentukan", "untuk", "kelas", "jika", "kegiatan", "seperti", "atau",
        "waktu", "bisa", "berbeda", "khusus", "yang", "kami", "ayah", "bunda",
        "sekolah", "mhis", "anak", "saya", "tahu", "jamnya", "berapa", "adalah",
        "informasi", "dalam", "buku", "panduan", "dengan", "dapat", "lebih", "lanjut"
    }
    filtered_words = [w for w in words if w not in stopwords]

    best_source = None
    best_page = None
    best_score = 0

    for s_name, pages in sources.items():
        for p_num, text in pages.items():
            t_norm = text.lower()
            score = 0
            for tm in time_matches:
                if tm in text or tm.replace(":", ".") in text or tm.replace(".", ":") in text:
                    score += 15
            for kw in filtered_words:
                if kw in t_norm:
                    score += 3

            if score > best_score:
                best_score = score
                best_source = s_name
                best_page = p_num

    if best_source and best_page and best_score >= 10:
        page_text = sources[best_source][best_page]
        lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
        matched_lines = []
        for line in lines:
            line_lower = line.lower()
            if any(tm in line for tm in time_matches) or any(
                kw in line_lower for kw in filtered_words if len(kw) >= 5
            ):
                matched_lines.append(line)
        snippet = "\n".join(matched_lines[:3]) if matched_lines else (lines[0] if lines else "Handbook reference")
        return [CitationOut(source=best_source, page=best_page, quote=snippet[:250])]

    return []


# ---------- Skema ----------
class CreateSessionIn(BaseModel):
    divisions: list[str] | str | None = Field(
        default=["general"],
        description="Daftar divisi anak (bisa lebih dari 1), contoh: ['preschool', 'primary'] atau ['all']"
    )
    division: str | None = Field(
        default=None,
        description="Alias tunggal jika hanya 1 divisi (misal: 'primary')"
    )


class SessionOut(BaseModel):
    session_id: str
    divisions: list[str]
    expires_in_seconds: int


class ModelCitation(BaseModel):
    source: str = Field(
        default="Primary",
        description="Nama dokumen handbook: 'Primary', 'Preschool', 'DC', 'Secondary', atau 'General'",
    )
    page: int = Field(
        ...,
        description="Nomor halaman persis tempat informasi ditemukan (lihat tag Page X)",
    )
    quote: str = Field(
        ...,
        description="Kutipan langsung kalimat atau baris tabel dari handbook pada halaman tersebut yang membuktikan jawaban",
    )


class ModelAnswer(BaseModel):
    found: bool = Field(
        ...,
        description="True jika informasi ada/dapat dijawab dari handbook, False jika tidak ada sama sekali",
    )
    answer: str = Field(
        ...,
        description="Jawaban ramah, empatik (Ayah/Bunda), akurat sesuai handbook dalam bahasa penanya",
    )
    citations: list[ModelCitation] = Field(
        default_factory=list,
        description="WAJIB diisi minimal 1 citation jika found=true. DILARANG KOSONG jika found=true!",
    )


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class CitationOut(BaseModel):
    source: str
    page: int
    quote: str


class AnswerOut(BaseModel):
    found: bool
    answer: str
    citations: list[CitationOut]


# ---------- Session store ----------
@dataclass
class Session:
    divisions: list[str] = field(default_factory=lambda: ["general"])
    history: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


SESSIONS: dict[str, Session] = {}


async def cleanup_loop():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        expired = [
            sid for sid, s in SESSIONS.items() if now - s.last_active > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            SESSIONS.pop(sid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

# Izinkan CORS agar bisa diakses oleh frontend/klien lain di jaringan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_token(x_app_token: str = Header(default="")):
    # Untuk produksi: ganti dengan validasi token login orang tua (JWT) dari sistem Anda.
    if not secrets.compare_digest(x_app_token, APP_TOKEN):
        raise HTTPException(status_code=401, detail="unauthorized")


# ---------- Endpoint ----------
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Handbook AI Chatbot API",
        "docs_url": "/docs"
    }


@app.post("/sessions", response_model=SessionOut)
async def create_session(body: CreateSessionIn | None = None, _: None = Depends(require_token)):
    if len(SESSIONS) >= MAX_SESSIONS:
        raise HTTPException(status_code=503, detail="too_many_sessions")
    sid = secrets.token_urlsafe(24)
    raw_div = body.division if (body and body.division) else (body.divisions if body else None)
    divs = parse_divisions(raw_div)
    SESSIONS[sid] = Session(divisions=divs)
    return SessionOut(session_id=sid, divisions=divs, expires_in_seconds=SESSION_TTL_SECONDS)


@app.post("/sessions/{session_id}/messages", response_model=AnswerOut)
async def send_message(session_id: str, body: AskIn, _: None = Depends(require_token)):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session_not_found")

    async with session.lock:  # cegah dua pertanyaan bersamaan merusak urutan riwayat
        session.last_active = time.time()
        system_prompt = build_system_prompt(session.divisions)
        messages = (
            [{"role": "system", "content": system_prompt}]
            + session.history
            + [{"role": "user", "content": body.question}]
        )
        try:
            resp = await client.responses.parse(
                model=MODEL,
                input=messages,
                text_format=ModelAnswer,
            )
            parsed = resp.output_parsed
        except Exception as e:
            print(f"[OpenAI Error] {type(e).__name__}: {e}")
            raise HTTPException(status_code=502, detail=f"ai_unavailable: {e}")

        verified = []
        if parsed and parsed.citations:
            for c in parsed.citations:
                c_src = getattr(c, "source", "Primary")
                res_source, res_page, res_quote = resolve_citation(
                    c.quote, c.page, c_src, session.divisions
                )
                verified.append(CitationOut(source=res_source, page=res_page, quote=res_quote))

        # Proteksi: jika model mengembalikan found=True tetapi citations kosong
        if parsed and parsed.found and not verified:
            active_srcs = get_active_sources(session.divisions)
            recovered = find_fallback_citations(parsed.answer, body.question, active_srcs)
            if recovered:
                verified.extend(recovered)

        # Penanganan hasil:
        if parsed and parsed.found:
            out = AnswerOut(
                found=True,
                answer=parsed.answer,
                citations=verified,
            )
        elif parsed and not parsed.found:
            # Model secara benar menyatakan info tidak ada (dijawab sopan sesuai bahasa pengguna)
            out = AnswerOut(found=False, answer=parsed.answer, citations=[])
        else:
            # Gagal parse total
            out = AnswerOut(found=False, answer=NOT_FOUND_MSG, citations=[])

        # Simpan hanya teks tanya-jawab (bukan JSON penuh) agar hemat token
        session.history += [
            {"role": "user", "content": body.question},
            {"role": "assistant", "content": out.answer},
        ]
        session.history = session.history[-MAX_HISTORY_MESSAGES:]
        session.last_active = time.time()

    return out


@app.delete("/sessions/{session_id}", status_code=204)
async def close_session(session_id: str, _: None = Depends(require_token)):
    SESSIONS.pop(session_id, None)  # idempotent: tidak error bila sudah tidak ada
    return Response(status_code=204)