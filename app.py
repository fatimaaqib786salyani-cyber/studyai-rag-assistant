import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np
import json
import re
import html


st.set_page_config(
    page_title="StudyAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root{
    --bg:#f5f5f7;
    --surface:#ffffff;
    --surface-2:#fbfbfd;
    --text:#1d1d1f;
    --secondary:#6e6e73;
    --tertiary:#86868b;
    --blue:#0071e3;
    --blue-soft:#e8f2ff;
    --green:#248a3d;
    --line:rgba(0,0,0,.075);
    --shadow:0 12px 40px rgba(0,0,0,.055);
}
html,body,[class*="css"]{
    font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
    "Helvetica Neue",Arial,sans-serif;
}
.stApp{background:var(--bg);color:var(--text);}
.main .block-container{
    max-width:940px;
    padding-top:2.35rem;
    padding-bottom:8rem;
}

/* Top identity */
.app-kicker{
    color:var(--tertiary);font-size:11px;font-weight:700;letter-spacing:.12em;
    margin-bottom:10px;
}
.app-title{
    margin:0;color:var(--text);font-size:2.65rem;line-height:1.04;
    font-weight:760;letter-spacing:-.055em;
}
.app-subtitle{
    max-width:650px;margin:11px 0 25px;color:var(--secondary);
    font-size:1rem;line-height:1.55;letter-spacing:-.01em;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:rgba(248,248,250,.96);
    border-right:1px solid var(--line);
}
[data-testid="stSidebar"]>div:first-child{padding:2rem 1.35rem;}
.brand-row{display:flex;align-items:center;gap:11px;margin-bottom:6px;}
.brand-mark{
    width:36px;height:36px;display:flex;align-items:center;justify-content:center;
    border-radius:11px;background:#1d1d1f;color:#fff;font-size:15px;font-weight:750;
    box-shadow:0 5px 16px rgba(0,0,0,.10);
}
.brand-name{font-size:20px;font-weight:750;letter-spacing:-.04em;color:var(--text);}
.brand-copy{margin:0 0 27px 47px;color:var(--tertiary);font-size:11px;}
.section-label{
    margin:0 0 9px;color:var(--tertiary);font-size:10px;font-weight:700;
    letter-spacing:.13em;
}
[data-testid="stFileUploaderDropzone"]{
    min-height:116px;padding:.75rem;background:#fff;
    border:1px dashed rgba(0,0,0,.14);border-radius:15px;
    transition:border-color .18s ease,box-shadow .18s ease,transform .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
    border-color:rgba(0,113,227,.45);box-shadow:0 7px 22px rgba(0,0,0,.045);
    transform:translateY(-1px);
}
[data-testid="stFileUploaderDropzone"] button{border-radius:9px;}
.stButton>button{
    min-height:42px;border:1px solid var(--line);border-radius:12px;
    background:#fff;color:var(--text);font-weight:560;
    transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease;
}
.stButton>button:hover{
    border-color:rgba(0,0,0,.13);box-shadow:0 6px 18px rgba(0,0,0,.055);
    transform:translateY(-1px);
}
.doc-card{
    margin-top:16px;padding:14px;border:1px solid var(--line);border-radius:16px;
    background:#fff;box-shadow:0 4px 18px rgba(0,0,0,.025);
}
.doc-top{display:flex;gap:10px;align-items:flex-start;}
.doc-icon{
    width:34px;height:34px;flex:0 0 34px;display:flex;align-items:center;
    justify-content:center;border-radius:10px;background:#eef4ff;color:#2b67b1;
    font-size:9px;font-weight:800;letter-spacing:.04em;
}
.doc-name{font-size:12px;line-height:1.35;font-weight:650;overflow-wrap:anywhere;}
.doc-meta{font-size:10px;color:var(--tertiary);margin-top:4px;}
.ready-row{
    display:flex;align-items:center;justify-content:space-between;margin-top:12px;
    padding-top:10px;border-top:1px solid rgba(0,0,0,.055);
}
.ready-pill{font-size:10px;color:var(--green);font-weight:700;}
.vector-copy{font-size:10px;color:var(--tertiary);}

/* Mode navigation */
.mode-label{
    color:var(--tertiary);font-size:10px;font-weight:700;letter-spacing:.12em;
    margin:0 0 8px;
}
div[role="radiogroup"]{
    display:flex!important;gap:6px!important;padding:5px!important;
    background:#e9e9ec!important;border-radius:14px!important;
    width:fit-content!important;margin-bottom:20px!important;
}
div[role="radiogroup"] label{
    margin:0!important;padding:8px 13px!important;border-radius:10px!important;
    transition:background .16s ease,box-shadow .16s ease!important;
}
div[role="radiogroup"] label:has(input:checked){
    background:#fff!important;box-shadow:0 1px 5px rgba(0,0,0,.10)!important;
}
div[role="radiogroup"] label>div:first-child{display:none!important;}
div[role="radiogroup"] p{
    font-size:13px!important;font-weight:570!important;color:#3a3a3c!important;
    white-space:nowrap!important;
}

/* Feature header */
.feature-card{
    display:flex;align-items:center;gap:15px;padding:20px 21px;
    border:1px solid var(--line);border-radius:20px;background:var(--surface);
    box-shadow:0 8px 30px rgba(0,0,0,.035);margin-bottom:13px;
}
.feature-icon{
    width:44px;height:44px;flex:0 0 44px;display:flex;align-items:center;
    justify-content:center;border-radius:13px;background:#f2f2f4;font-size:20px;
}
.feature-title{
    color:var(--text);font-size:18px;font-weight:720;letter-spacing:-.025em;
}
.feature-copy{color:var(--secondary);font-size:12px;line-height:1.45;margin-top:3px;}

/* Generated content */
.result-shell{
    padding:4px 2px 0;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3{
    color:var(--text);letter-spacing:-.025em;
}
[data-testid="stMarkdownContainer"] h3{
    margin-top:1.35rem;font-size:1.22rem;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{
    color:#303033;line-height:1.65;
}
[data-testid="stMarkdownContainer"] hr{
    border:0;border-top:1px solid var(--line);margin:1.6rem 0;
}

/* Empty / Ask mode */
.empty-state{
    padding:2.6rem 1.8rem;border:1px solid var(--line);border-radius:20px;
    background:#fff;text-align:center;box-shadow:0 8px 30px rgba(0,0,0,.03);
}
.empty-icon{
    width:44px;height:44px;margin:0 auto 13px;display:flex;align-items:center;
    justify-content:center;border-radius:14px;background:#f2f2f4;font-size:18px;
}
.empty-title{font-size:17px;font-weight:700;letter-spacing:-.025em;}
.empty-copy{
    max-width:470px;margin:6px auto 0;color:var(--secondary);font-size:13px;line-height:1.5;
}

/* Chat */
[data-testid="stChatMessage"]{background:transparent;border:0;padding:.8rem 0;margin:0;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
    margin:1rem 0 .25rem;padding:.85rem 1rem;background:#e9e9ec;
    border-radius:16px;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){
    padding:1rem .25rem 1.25rem;border-bottom:1px solid var(--line);
}
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li{
    font-size:.98rem;line-height:1.65;color:var(--text);
}
.source-chip{
    display:inline-flex;align-items:center;margin-top:7px;padding:5px 9px;
    border-radius:999px;background:#e9e9ec;color:#6e6e73;font-size:10px;font-weight:650;
}
[data-testid="stExpander"]{
    margin-top:10px;border:1px solid var(--line);border-radius:13px;background:#fff;
}
[data-testid="stChatInput"]{
    max-width:940px;margin:0 auto 18px;background:#fff;border:1px solid rgba(0,0,0,.10);
    border-radius:16px;box-shadow:0 10px 32px rgba(0,0,0,.09);
}
[data-testid="stChatInput"]:focus-within{
    border-color:rgba(0,113,227,.42);box-shadow:0 10px 34px rgba(0,0,0,.11);
}

/* Captions + chrome */
[data-testid="stCaptionContainer"]{color:var(--tertiary);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stHeader"]{
    background:rgba(245,245,247,.86);backdrop-filter:blur(18px);
}

/* Mobile */
@media(max-width:768px){
    .main .block-container{padding:1.7rem 1rem 7.5rem;}
    .app-title{font-size:2.25rem;}
    .feature-card{padding:16px;}
    div[role="radiogroup"]{width:100%!important;overflow-x:auto!important;}
    div[role="radiogroup"] label{padding:7px 10px!important;}
}

/* Premium study components */

[data-testid="column"] .flash-card{margin-bottom:14px;height:calc(100% - 14px);}

.study-grid{
    display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:14px 0 20px;
}
.study-stat{
    padding:16px;border:1px solid var(--line);border-radius:16px;background:#fff;
}
.study-stat-label{font-size:10px;font-weight:700;letter-spacing:.09em;color:var(--tertiary);}
.study-stat-value{margin-top:5px;font-size:18px;font-weight:720;letter-spacing:-.025em;color:var(--text);}

.flash-card{
    position:relative;min-height:190px;padding:20px;border:1px solid var(--line);
    border-radius:20px;background:#fff;box-shadow:0 8px 28px rgba(0,0,0,.035);
    transition:transform .18s ease,box-shadow .18s ease;
}
.flash-card:hover{transform:translateY(-2px);box-shadow:0 14px 34px rgba(0,0,0,.07);}
.flash-number{
    width:30px;height:30px;display:flex;align-items:center;justify-content:center;
    border-radius:10px;background:#f2f2f4;color:var(--secondary);font-size:11px;font-weight:750;
}
.flash-question{
    margin-top:17px;color:var(--text);font-size:16px;line-height:1.38;font-weight:700;
    letter-spacing:-.02em;
}
.flash-divider{height:1px;background:var(--line);margin:16px 0 13px;}
.flash-answer{color:var(--secondary);font-size:13px;line-height:1.55;}
.flash-answer strong{color:var(--text);font-weight:650;}

.summary-hero{
    padding:22px;border:1px solid var(--line);border-radius:20px;background:#fff;
    box-shadow:0 8px 30px rgba(0,0,0,.03);margin:14px 0 18px;
}
.summary-kicker{font-size:10px;font-weight:750;letter-spacing:.1em;color:var(--blue);}
.summary-title{font-size:21px;font-weight:740;letter-spacing:-.035em;margin-top:7px;}
.summary-copy{font-size:13px;line-height:1.55;color:var(--secondary);margin-top:5px;}

.quiz-card{
    padding:20px 20px 8px;border:1px solid var(--line);border-radius:20px;
    background:#fff;box-shadow:0 8px 28px rgba(0,0,0,.03);margin:14px 0;
}
.quiz-number{
    display:inline-flex;padding:5px 8px;border-radius:8px;background:#f2f2f4;
    color:var(--secondary);font-size:10px;font-weight:750;letter-spacing:.05em;
}
.quiz-question{font-size:16px;font-weight:700;line-height:1.45;letter-spacing:-.018em;margin:11px 0 5px;}
.quiz-score{
    display:flex;align-items:center;justify-content:space-between;padding:17px 19px;
    border-radius:17px;background:#1d1d1f;color:#fff;margin:16px 0;
}
.quiz-score strong{font-size:20px;letter-spacing:-.03em;}
.quiz-score span{font-size:12px;color:#d2d2d7;}

.quick-label{font-size:10px;font-weight:700;letter-spacing:.1em;color:var(--tertiary);margin:18px 0 8px;}
.ask-welcome{
    padding:22px;border:1px solid var(--line);border-radius:20px;background:#fff;
    box-shadow:0 8px 30px rgba(0,0,0,.03);margin-bottom:13px;
}
.ask-title{font-size:19px;font-weight:730;letter-spacing:-.03em;}
.ask-copy{font-size:13px;color:var(--secondary);line-height:1.5;margin-top:5px;}

@media(max-width:768px){
    .study-grid{grid-template-columns:1fr;}
    .flash-card{min-height:auto;}
}


/* ===== Final layout polish ===== */
.main .block-container{
    max-width:960px!important;
    padding-top:2.2rem!important;
    padding-left:2rem!important;
    padding-right:2rem!important;
    padding-bottom:8rem!important;
}
.app-kicker{margin-bottom:12px!important;}
.app-title{max-width:760px;}
.app-subtitle{max-width:720px;margin-bottom:28px!important;}

[data-testid="stSidebar"]{min-width:330px!important;max-width:330px!important;}
[data-testid="stSidebar"]>div:first-child{padding:2rem 1.45rem 2rem!important;}
.sidebar-divider{height:1px;background:var(--line);margin:18px 0;}
.sidebar-section{margin-top:18px;}
.sidebar-info{
    padding:14px;border:1px solid var(--line);border-radius:15px;background:rgba(255,255,255,.76);
}
.sidebar-info-row{
    display:flex;align-items:flex-start;justify-content:space-between;gap:12px;
    padding:8px 0;border-bottom:1px solid rgba(0,0,0,.05);
}
.sidebar-info-row:last-child{border-bottom:0;padding-bottom:0;}
.sidebar-info-row:first-child{padding-top:0;}
.sidebar-info-label{font-size:10px;color:var(--tertiary);font-weight:650;}
.sidebar-info-value{font-size:10px;color:var(--text);font-weight:700;text-align:right;}
.privacy-card{
    margin-top:14px;padding:13px 14px;border-radius:14px;background:#eef6ff;
    border:1px solid rgba(0,113,227,.08);
}
.privacy-title{font-size:11px;color:#245d9c;font-weight:700;}
.privacy-copy{font-size:10px;color:#5f7590;line-height:1.45;margin-top:4px;}

.mode-label{margin-top:3px!important;margin-bottom:9px!important;}
div[role="radiogroup"]{margin-bottom:26px!important;}

.feature-card{margin-bottom:15px!important;}
.study-grid{margin-top:0!important;margin-bottom:14px!important;}
.study-stat{min-height:84px;display:flex;flex-direction:column;justify-content:center;}
.summary-hero{margin-top:18px!important;margin-bottom:22px!important;}

/* Keep generated study guide typography proportional to the product UI */
[data-testid="stMarkdownContainer"] h1{
    font-size:2rem!important;line-height:1.12!important;letter-spacing:-.04em!important;
    margin-top:1.65rem!important;margin-bottom:.7rem!important;
}
[data-testid="stMarkdownContainer"] h2{
    font-size:1.35rem!important;line-height:1.2!important;letter-spacing:-.025em!important;
    margin-top:1.7rem!important;margin-bottom:.55rem!important;
}
[data-testid="stMarkdownContainer"] h3{
    font-size:1.08rem!important;margin-top:1.35rem!important;margin-bottom:.45rem!important;
}
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol{padding-left:1.25rem!important;}
[data-testid="stMarkdownContainer"] li{margin-bottom:.28rem!important;}

[data-testid="stButton"]{margin-top:2px;}
[data-testid="stButton"] button{font-size:13px!important;}
[data-testid="stFileUploader"]{margin-bottom:4px;}
[data-testid="stFileUploaderFile"]{padding-top:8px!important;padding-bottom:8px!important;}

.flash-card{min-height:205px!important;}
.quiz-card{margin-top:16px!important;margin-bottom:9px!important;}

/* Footer/status strip */
.workspace-foot{
    margin-top:18px;padding-top:13px;border-top:1px solid var(--line);
    display:flex;align-items:center;justify-content:space-between;gap:12px;
    color:var(--tertiary);font-size:10px;
}
.workspace-foot strong{color:var(--secondary);font-weight:650;}

@media(max-width:900px){
    [data-testid="stSidebar"]{min-width:300px!important;max-width:300px!important;}
}
@media(max-width:768px){
    .main .block-container{
        padding-top:1.5rem!important;padding-left:1rem!important;
        padding-right:1rem!important;padding-bottom:7.5rem!important;
    }
    .app-title{font-size:2.15rem!important;}
    .app-subtitle{font-size:.95rem!important;}
}


/* ===== Advanced Apple-inspired product layer ===== */
.stApp{
 background:
 radial-gradient(circle at 82% 4%,rgba(41,151,255,.10),transparent 25rem),
 radial-gradient(circle at 16% 34%,rgba(125,92,255,.055),transparent 27rem),
 linear-gradient(180deg,#fafafe 0%,#f3f3f6 52%,#f8f8fa 100%);
}
.main .block-container{max-width:1040px!important;padding-top:2.7rem!important;}
[data-testid="stSidebar"]{background:rgba(247,247,250,.84)!important;backdrop-filter:blur(28px) saturate(150%);-webkit-backdrop-filter:blur(28px) saturate(150%);box-shadow:12px 0 40px rgba(0,0,0,.025);}
[data-testid="stSidebar"]>div:first-child{padding:1.8rem 1.35rem 2.2rem!important;}
.brand-mark{background:linear-gradient(145deg,#111,#343438)!important;border-radius:12px!important;}
.brand-copy{line-height:1.5!important;max-width:215px;}
.sidebar-stepper{display:grid;gap:8px;margin:11px 0 5px;}
.sidebar-step{display:flex;gap:10px;align-items:flex-start;padding:10px 11px;border-radius:13px;background:rgba(255,255,255,.64);border:1px solid rgba(0,0,0,.045);}
.step-dot{width:22px;height:22px;flex:0 0 22px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#1d1d1f;color:#fff;font-size:9px;font-weight:800;}
.step-title{font-size:10px;font-weight:750;color:#2c2c2e}.step-copy{font-size:9px;line-height:1.4;color:#86868b;margin-top:2px;}
.sidebar-help{margin-top:14px;padding:14px;border-radius:16px;background:rgba(255,255,255,.68);border:1px solid rgba(0,0,0,.055);}
.sidebar-help-title{font-size:11px;font-weight:750;margin-bottom:5px}.sidebar-help-copy{font-size:10px;color:#737378;line-height:1.5;}
.hero-shell{position:relative;overflow:hidden;padding:37px 39px 35px;margin-bottom:27px;border:1px solid rgba(255,255,255,.75);border-radius:30px;background:linear-gradient(145deg,rgba(255,255,255,.88),rgba(250,250,253,.72));box-shadow:0 24px 70px rgba(0,0,0,.055);backdrop-filter:blur(20px);}
.hero-shell:after{content:"";position:absolute;width:310px;height:310px;border-radius:50%;right:-115px;top:-165px;background:radial-gradient(circle,rgba(41,151,255,.18),rgba(125,92,255,.045) 48%,transparent 70%);}
.hero-kicker{font-size:10px;font-weight:800;letter-spacing:.15em;color:#6e6e73;margin-bottom:14px}.hero-title{position:relative;z-index:1;margin:0;max-width:780px;font-size:3.3rem;line-height:1.01;letter-spacing:-.065em;font-weight:780}.hero-copy{position:relative;z-index:1;margin:16px 0 0;max-width:700px;color:#66666b;font-size:1.02rem;line-height:1.6;}
.hero-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}.hero-chip{padding:7px 10px;border-radius:999px;background:rgba(242,242,245,.92);font-size:10px;font-weight:700;color:#6e6e73;border:1px solid rgba(0,0,0,.035);}
.workflow-card{margin:2px 0 25px;padding:17px 19px;border-radius:19px;background:rgba(255,255,255,.67);border:1px solid rgba(0,0,0,.055)}.workflow-title{font-size:10px;font-weight:780;letter-spacing:.1em;color:#86868b}.workflow-row{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-top:11px}.workflow-node{padding:7px 9px;border-radius:10px;background:#fff;border:1px solid rgba(0,0,0,.055);font-size:10px;font-weight:700}.workflow-arrow{color:#a1a1a6;font-size:11px}
.feature-card,.summary-hero,.ask-welcome,.flash-card,.quiz-card,.empty-state,.study-stat{background:rgba(255,255,255,.86)!important;backdrop-filter:blur(15px)}
.feature-card{border-radius:24px!important;padding:22px 23px!important;box-shadow:0 14px 44px rgba(0,0,0,.045)!important}.flash-card{border-radius:23px!important;box-shadow:0 12px 34px rgba(0,0,0,.035)!important}.flash-card:hover{transform:translateY(-4px)!important;box-shadow:0 20px 45px rgba(0,0,0,.075)!important}.study-stat{border-radius:19px!important;box-shadow:0 10px 30px rgba(0,0,0,.025)}
[data-testid="stButton"] button{border-radius:13px!important;transition:all .18s ease!important}div[role="radiogroup"]{background:rgba(229,229,234,.76)!important;backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.72)}
.workspace-foot{margin-top:34px!important;padding:18px 2px!important}
@media(max-width:768px){.hero-shell{padding:26px 22px;border-radius:24px}.hero-title{font-size:2.42rem}.hero-copy{font-size:.95rem}.main .block-container{padding-top:1.25rem!important}}

</style>
""", unsafe_allow_html=True)

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded notes."

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

for key, default in {
    "messages": [],
    "current_file": None,
    "summary": None,
    "quiz": None,
    "flashcards": None,
    "quiz_answers": {},
    "quiz_submitted": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def create_chunks(text, chunk_size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def source_badge(pages):
    pages = sorted(set(pages))
    label = "Page" if len(pages) == 1 else "Pages"
    page_text = ", ".join(map(str, pages))
    st.markdown(
        f'<div class="source-chip">Source · {label} {page_text}</div>',
        unsafe_allow_html=True,
    )

def ask_llm(prompt, temperature=0):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "I couldn't generate a response right now. Please try again in a moment."

def retrieve(query, documents, index, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, min(top_k, len(documents)))

    results = []
    for rank, doc_index in enumerate(indices[0]):
        if doc_index == -1:
            continue
        item = documents[doc_index].copy()
        item["score"] = float(scores[0][rank])
        results.append(item)
    return results

def build_context(items):
    return "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}" for item in items
    )

def whole_document_context(documents):
    return "\n\n".join(
        f"[Page {doc['page']}]\n{doc['text']}" for doc in documents
    )


def clean_json_response(text):
    """Extract JSON even if the model wraps it in a markdown code fence."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start_candidates = [p for p in (cleaned.find("["), cleaned.find("{")) if p != -1]
    if start_candidates:
        cleaned = cleaned[min(start_candidates):]
    return cleaned

def parse_flashcards(text):
    try:
        data = json.loads(clean_json_response(text))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def parse_quiz(text):
    try:
        data = json.loads(clean_json_response(text))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def render_flashcards(cards):
    if not cards:
        st.warning("Flashcards could not be formatted. Please generate them again.")
        return

    # Render each card separately. This avoids Streamlit Markdown treating
    # multi-line HTML as a code block and keeps the two-column layout stable.
    rows = [cards[i:i + 2] for i in range(0, len(cards), 2)]

    for row_start, row in enumerate(rows):
        columns = st.columns(2, gap="medium")

        for offset, card in enumerate(row):
            card_number = row_start * 2 + offset + 1
            question = html.escape(str(card.get("question", "")).strip())
            answer = html.escape(str(card.get("answer", "")).strip())

            with columns[offset]:
                card_html = (
                    f'<div class="flash-card">'
                    f'<div class="flash-number">{card_number:02d}</div>'
                    f'<div class="flash-question">{question}</div>'
                    f'<div class="flash-divider"></div>'
                    f'<div class="flash-answer"><strong>Answer</strong><br>{answer}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div class="brand-row">
        <div class="brand-mark">S</div>
        <div class="brand-name">StudyAI</div>
    </div>
    <div class="brand-copy">Your private, document-grounded study workspace.</div>

    <div class="sidebar-divider"></div>
    <div class="section-label">STUDY DOCUMENT</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF Notes", type=["pdf"], label_visibility="collapsed"
    )

    st.markdown("""
    <div class="privacy-card">
        <div class="privacy-title">Grounded by design</div>
        <div class="privacy-copy">
            Answers, summaries, quizzes and flashcards are generated from
            the study material you upload.
        </div>
    </div>
    <div class="sidebar-section">
        <div class="section-label">HOW IT WORKS</div>
        <div class="sidebar-stepper">
          <div class="sidebar-step"><div class="step-dot">1</div><div><div class="step-title">Upload your notes</div><div class="step-copy">Add a selectable-text PDF from your course material.</div></div></div>
          <div class="sidebar-step"><div class="step-dot">2</div><div><div class="step-title">Choose a study mode</div><div class="step-copy">Ask, summarize, test yourself, or create revision cards.</div></div></div>
          <div class="sidebar-step"><div class="step-dot">3</div><div><div class="step-title">Study with evidence</div><div class="step-copy">Grounded answers can be checked against retrieved passages.</div></div></div>
        </div>
    </div>
    <div class="sidebar-section"><div class="section-label">WORKSPACE</div></div>
    """, unsafe_allow_html=True)

    if st.button("Clear workspace", use_container_width=True):
        st.session_state.messages = []
        st.session_state.summary = None
        st.session_state.quiz = None
        st.session_state.flashcards = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.rerun()

st.markdown("""
<div class="hero-shell">
 <div class="hero-kicker">STUDYAI · DOCUMENT INTELLIGENCE</div>
 <h1 class="hero-title">Turn your notes into an active study workspace.</h1>
 <p class="hero-copy">Upload one PDF, then move naturally between grounded Q&amp;A, a structured study guide, self-testing, and fast revision cards — all from the same material.</p>
 <div class="hero-chips"><span class="hero-chip">Notes-only answers</span><span class="hero-chip">Source-aware retrieval</span><span class="hero-chip">Interactive practice</span><span class="hero-chip">Private workspace</span></div>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📄</div>
        <div class="empty-title">Bring your notes to life.</div>
        <div class="empty-copy">Upload a PDF from the sidebar to begin.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

file_identity = (uploaded_file.name, uploaded_file.size)
if st.session_state.current_file != file_identity:
    st.session_state.current_file = file_identity
    st.session_state.messages = []
    st.session_state.summary = None
    st.session_state.quiz = None
    st.session_state.flashcards = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False

reader = PdfReader(uploaded_file)
documents = []

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()
    if page_text:
        for chunk_number, chunk in enumerate(create_chunks(page_text), start=1):
            documents.append({
                "text": chunk,
                "page": page_number,
                "chunk": chunk_number,
            })

if not documents:
    st.error("No readable text was found in this PDF.")
    st.stop()

chunk_texts = [doc["text"] for doc in documents]
with st.spinner("Preparing your notes..."):
    embeddings = model.encode(chunk_texts, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="section-label">CURRENT DOCUMENT</div>
        <div class="doc-card">
            <div class="doc-top">
                <div class="doc-icon">PDF</div>
                <div>
                    <div class="doc-name">{uploaded_file.name}</div>
                    <div class="doc-meta">Processed and ready for study</div>
                </div>
            </div>
            <div class="ready-row">
                <div class="ready-pill">● &nbsp;Ready</div>
                <div class="vector-copy">{index.ntotal} vectors indexed</div>
            </div>
        </div>
        <div class="sidebar-info">
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Pages</span>
                <span class="sidebar-info-value">{len(reader.pages)}</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Text chunks</span>
                <span class="sidebar-info-value">{len(documents)}</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Vector search</span>
                <span class="sidebar-info-value">FAISS</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Answer policy</span>
                <span class="sidebar-info-value">Notes only</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True) 
    st.markdown("""
    <div class="sidebar-help">
      <div class="sidebar-help-title">Tips for stronger results</div>
      <div class="sidebar-help-copy">Use text-based PDFs, ask specific questions, and open supporting passages whenever you want to verify how an answer was grounded.</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="mode-label">CHOOSE A MODE</div>', unsafe_allow_html=True)

# Streamlit 1.31 does not have segmented_control, so use horizontal radio.
mode = st.radio(
    "Study Mode",
    ["Ask Notes", "Summary", "Quiz", "Flashcards"],
    horizontal=True,
    label_visibility="collapsed",
)

mode_steps = {
    "Ask Notes": ["Ask a question", "Retrieve passages", "Grounded answer", "Verify sources"],
    "Summary": ["Read document", "Organize concepts", "Build study guide", "Quick revision"],
    "Quiz": ["Generate questions", "Choose answers", "Submit", "Score + explanations"],
    "Flashcards": ["Find key ideas", "Create cards", "Review concepts", "Revise faster"],
}
st.markdown(
    '<div class="workflow-card"><div class="workflow-title">' + mode.upper()
    + ' · WHAT HAPPENS</div><div class="workflow-row">'
    + '<span class="workflow-arrow">→</span>'.join(
        f'<span class="workflow-node">{step}</span>' for step in mode_steps[mode]
    ) + '</div></div>', unsafe_allow_html=True
)

if mode == "Ask Notes":
    if not st.session_state.messages:
        st.markdown("""
        <div class="ask-welcome">
            <div class="ask-title">What would you like to understand?</div>
            <div class="ask-copy">
                Ask a direct question from your notes. StudyAI finds the most relevant
                passages first, then answers only from that evidence.
            </div>
        </div>
        <div class="quick-label">TRY A QUESTION</div>
        """, unsafe_allow_html=True)

        qcols = st.columns(3)
        suggestions = [
            "Explain the main concept",
            "What should I remember?",
            "Give me a simple explanation",
        ]
        for col, suggestion in zip(qcols, suggestions):
            with col:
                if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                source_badge(message["sources"])
            if message["role"] == "assistant" and message.get("retrieved"):
                with st.expander("View supporting passages"):
                    for doc in message["retrieved"]:
                        st.markdown(f"**Page {doc['page']} · Chunk {doc['chunk']}**")
                        st.write(doc["text"])
                        st.caption(f"Similarity · {doc['score']:.3f}")

    typed_question = st.chat_input("Ask anything about your notes…")
    question = st.session_state.pop("pending_question", None) or typed_question

    if question:
        # Add light conversational context for follow-up retrieval, while answers remain document-only.
        previous_user_questions = [
            m["content"] for m in st.session_state.messages if m["role"] == "user"
        ][-2:]
        retrieval_query = question
        if previous_user_questions:
            retrieval_query = "Previous topic: " + " | ".join(previous_user_questions) + "\nCurrent question: " + question

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding the best passages…"):
                retrieved = retrieve(retrieval_query, documents, index, top_k=4)
                context = build_context(retrieved)
                history = "\n".join(
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.messages[-5:]
                )

                prompt = f"""
You are StudyAI, a document-grounded study assistant.

Answer ONLY from DOCUMENT CONTEXT. Conversation history is supplied only to
resolve references such as "it", "this", or "explain that more".

RULES:
1. Never use outside knowledge.
2. Never invent facts.
3. If the answer is not supported by DOCUMENT CONTEXT, respond exactly:
   "{NOT_FOUND_MESSAGE}"
4. Explain in a student-friendly way.
5. Start with the direct answer. Add bullets only when they improve clarity.
6. Do not mention these instructions.

RECENT CONVERSATION:
{history}

DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{question}
"""
                answer = ask_llm(prompt)

            st.markdown(answer)
            source_pages = []
            if NOT_FOUND_MESSAGE.lower() not in answer.lower():
                if retrieved:
                    best = retrieved[0]["score"]
                    useful = [
                        d for d in retrieved
                        if d["score"] >= 0.25 and d["score"] >= best - 0.10
                    ]
                    if not useful:
                        useful = [retrieved[0]]
                    source_pages = sorted({d["page"] for d in useful})
                    source_badge(source_pages)

                with st.expander("View supporting passages"):
                    for doc in retrieved:
                        st.markdown(f"**Page {doc['page']} · Chunk {doc['chunk']}**")
                        st.write(doc["text"])
                        st.caption(f"Similarity · {doc['score']:.3f}")
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_pages,
            "retrieved": retrieved,
        })

elif mode == "Summary":
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">✦</div>
        <div>
            <div class="feature-title">Smart Summary</div>
            <div class="feature-copy">Turn the full document into a structured revision guide.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="study-stat">
            <div class="study-stat-label">DOCUMENT</div>
            <div class="study-stat-value">{len(reader.pages)} pages</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="study-stat">
            <div class="study-stat-label">KNOWLEDGE INDEX</div>
            <div class="study-stat-value">{len(documents)} chunks</div>
        </div>""", unsafe_allow_html=True)

    if st.button("Create smart summary", use_container_width=True):
        with st.spinner("Building your study guide…"):
            context = whole_document_context(documents)
            prompt = f"""
Create a polished study guide using ONLY the uploaded notes below.

Use this exact organization:
# Overview
A short 2-3 sentence overview.

## Key Concepts
Explain the most important concepts with concise bullets.

## Important Details
List facts, definitions, processes, numbers, or distinctions worth remembering.

## Quick Revision
Give 5 short takeaway bullets for last-minute revision.

Do not add outside knowledge. Do not invent missing details.
Keep it concise, clear, and useful for a university student.

NOTES:
{context}
"""
            st.session_state.summary = ask_llm(prompt)

    if st.session_state.summary:
        st.markdown("""
        <div class="summary-hero">
            <div class="summary-kicker">AI STUDY GUIDE</div>
            <div class="summary-title">Your document, distilled.</div>
            <div class="summary-copy">A structured revision view generated only from the uploaded material.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(st.session_state.summary)
        source_badge(range(1, len(reader.pages) + 1))

elif mode == "Quiz":
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">✓</div>
        <div>
            <div class="feature-title">Interactive Quiz</div>
            <div class="feature-copy">Answer first. Reveal your score and explanations when you submit.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Create a new quiz", use_container_width=True):
        with st.spinner("Designing your quiz…"):
            context = whole_document_context(documents)
            prompt = f"""
Using ONLY the notes below, return a valid JSON array with exactly 5 quiz objects.
No markdown and no text outside JSON.

Each object must have:
"question": string,
"options": array of exactly 4 concise strings,
"answer": integer from 0 to 3 indicating the correct option,
"explanation": one short sentence grounded in the notes.

Make distractors plausible. Do not use outside knowledge.
Vary the concepts tested.

NOTES:
{context}
"""
            raw_quiz = ask_llm(prompt, temperature=0.15)
            st.session_state.quiz = parse_quiz(raw_quiz)
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False

    quiz = st.session_state.quiz
    if quiz:
        letters = ["A", "B", "C", "D"]
        for i, item in enumerate(quiz):
            st.markdown(f"""
            <div class="quiz-card">
                <span class="quiz-number">QUESTION {i+1:02d}</span>
                <div class="quiz-question">{item.get("question","")}</div>
            </div>
            """, unsafe_allow_html=True)
            options = item.get("options", [])[:4]
            choice = st.radio(
                f"Answer question {i+1}",
                range(len(options)),
                format_func=lambda x, opts=options: f"{letters[x]}. {opts[x]}",
                index=None,
                key=f"quiz_choice_{i}",
                label_visibility="collapsed",
            )
            st.session_state.quiz_answers[i] = choice

        if st.button("Check my answers", type="primary", use_container_width=True):
            st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            score = 0
            for i, item in enumerate(quiz):
                selected = st.session_state.quiz_answers.get(i)
                correct = int(item.get("answer", 0))
                if selected == correct:
                    score += 1

            st.markdown(f"""
            <div class="quiz-score">
                <div><strong>{score} / {len(quiz)}</strong><br><span>Your quiz score</span></div>
                <div><span>{round((score/len(quiz))*100)}% complete</span></div>
            </div>
            """, unsafe_allow_html=True)

            for i, item in enumerate(quiz):
                selected = st.session_state.quiz_answers.get(i)
                correct = int(item.get("answer", 0))
                if selected == correct:
                    st.success(f"Question {i+1}: Correct — {item.get('explanation','')}")
                else:
                    answer_text = item.get("options", [""])[correct]
                    st.error(f"Question {i+1}: Correct answer: {answer_text}")
                    st.caption(item.get("explanation", ""))

elif mode == "Flashcards":
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">▣</div>
        <div>
            <div class="feature-title">Revision Cards</div>
            <div class="feature-copy">The most useful concepts, turned into clean scan-friendly cards.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Create revision cards", use_container_width=True):
        with st.spinner("Building your revision deck…"):
            context = whole_document_context(documents)
            prompt = f"""
Using ONLY the notes below, return a valid JSON array with exactly 8 objects.
No markdown and no text outside JSON.

Each object:
{{"question":"concise study question","answer":"brief grounded answer"}}

Select important definitions, concepts, comparisons, processes, and facts.
Do not use outside knowledge.

NOTES:
{context}
"""
            raw_cards = ask_llm(prompt, temperature=0.1)
            st.session_state.flashcards = parse_flashcards(raw_cards)

    if st.session_state.flashcards:
        st.markdown(
            f'<div class="quick-label">{len(st.session_state.flashcards)} CARDS · REVISION DECK</div>',
            unsafe_allow_html=True,
        )
        render_flashcards(st.session_state.flashcards)


st.markdown("""
<div class="workspace-foot">
    <span><strong>StudyAI</strong> · Document-grounded learning workspace</span>
    <span>PDF → Embeddings → FAISS → AI</span>
</div>
""", unsafe_allow_html=True)
