"""
app.py  —  Resume2Vec  |  Streamlit Web UI
Tabs:
  1. Optimize Resume  — analyze & improve existing resume
  2. Generate Resume  — build a new AI-generated resume from scratch (T5 GenAI)
"""

import os
import sys
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.resume_parser    import parse_resume
from src.ats_checker      import ats_score
from src.resume_optimizer import ResumeOptimizer, _t5_is_ready, T5_GEN_CKPT
from src.resume_generator import ResumeGenerator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume2Vec — ATS Optimizer",
    page_icon="📄",
    layout="wide",
)

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading optimizer model…")
def load_optimizer():
    return ResumeOptimizer()

@st.cache_resource(show_spinner="Loading generator model…")
def load_generator():
    return ResumeGenerator()

optimizer = load_optimizer()
generator = load_generator()

# ── Load Siamese model if available ──────────────────────────────────────────
siamese_model     = None
siamese_tokenizer = None
SIAMESE_CKPT      = os.path.join("models", "siamese_checkpoint.pt")

if os.path.exists(SIAMESE_CKPT):
    @st.cache_resource(show_spinner="Loading Siamese model…")
    def _load_siamese():
        from src.siamese_model import load_model
        return load_model()
    siamese_model, siamese_tokenizer = _load_siamese()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 Resume2Vec — ATS Resume Optimizer & Generator")
st.caption("Siamese Transformer  •  T5 Generative AI  •  ATS Scoring")

# ── Model status ──────────────────────────────────────────────────────────────
with st.expander("🤖 Model Status", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists(SIAMESE_CKPT):
            st.success("✅ Siamese Transformer")
        else:
            st.warning("⏳ Siamese — not trained")
    with c2:
        if _t5_is_ready():
            st.success("✅ T5 Optimizer")
        else:
            st.warning("⏳ T5 Optimizer — rule-based fallback")
    with c3:
        if _t5_is_ready(T5_GEN_CKPT):
            st.success("✅ T5 Generator (GenAI)")
        else:
            st.warning("⏳ T5 Generator — template fallback\n`python train_all.py --only-generator`")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_optimize, tab_generate = st.tabs([
    "🔍 Optimize Existing Resume",
    "✍️ Generate Resume with AI"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OPTIMIZE
# ─────────────────────────────────────────────────────────────────────────────
with tab_optimize:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Your Resume")
        input_mode = st.radio("Input method",
                              ["Paste text", "Upload PDF", "Upload DOCX"],
                              horizontal=True, key="opt_input_mode")
        resume_text = ""

        if input_mode == "Paste text":
            resume_text = st.text_area("Paste your resume here", height=300,
                                       placeholder="John Doe\nSoftware Engineer\n…",
                                       key="opt_paste")
        elif input_mode == "Upload PDF":
            up = st.file_uploader("Upload PDF", type=["pdf"], key="opt_pdf")
            if up:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up.read()); tmp_path = tmp.name
                resume_text = parse_resume(tmp_path, "pdf")
                st.text_area("Extracted text", resume_text[:800], height=200, key="opt_pdf_prev")
        elif input_mode == "Upload DOCX":
            up = st.file_uploader("Upload DOCX", type=["docx"], key="opt_docx")
            if up:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(up.read()); tmp_path = tmp.name
                resume_text = parse_resume(tmp_path, "docx")
                st.text_area("Extracted text", resume_text[:800], height=200, key="opt_docx_prev")

    with col_right:
        st.subheader("Job Description")
        jd_text = st.text_area("Paste the job description", height=300,
                                placeholder="We are looking for a Senior Python Developer…",
                                key="opt_jd")

    if st.button("🔍 Analyze & Optimize", type="primary",
                 use_container_width=True, key="btn_optimize"):
        if not resume_text.strip():
            st.error("Please provide your resume.")
        elif not jd_text.strip():
            st.error("Please provide the job description.")
        else:
            with st.spinner("Analyzing…"):
                report = ats_score(resume_text, jd_text,
                                   siamese_model=siamese_model,
                                   siamese_tokenizer=siamese_tokenizer)

            st.subheader("📊 ATS Compatibility Report")
            score = report["overall_score"]
            color = "🟢" if score >= 70 else "🟡" if score >= 45 else "🔴"
            st.metric(f"{color} Overall ATS Score", f"{score} / 100")
            st.progress(int(score))

            b = report["breakdown"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Keywords",  f"{b['keyword']} pts")
            c2.metric("Sections",  f"{b['sections']} pts")
            c3.metric("Format",    f"{b['format']} pts")
            c4.metric("Semantic",  f"{b['semantic']} pts")

            kw = report["keyword_coverage"]
            with st.expander(f"🔑 Keyword Coverage  {kw['coverage_pct']}%"):
                st.write("**Matched:**", ", ".join(kw["matched"]) or "—")
                st.write("**Missing:**", ", ".join(kw["missing"]) or "—")

            with st.expander("📋 Sections"):
                for sec, found in report["sections"].items():
                    st.write(("✅" if found else "❌") + f"  {sec.capitalize()}")

            if report["format_warnings"]:
                with st.expander("⚠️ Format Warnings"):
                    for w in report["format_warnings"]:
                        st.warning(w)

            st.divider()
            st.subheader("✨ AI-Optimized Resume")
            mode_label = "T5 Generative AI" if _t5_is_ready() else "Rule-based keyword optimizer"
            st.caption(f"Using: {mode_label}")

            with st.spinner("Optimizing…"):
                optimized = optimizer.optimize(resume_text, jd_text)

            st.text_area("Optimized resume:", optimized, height=350, key="opt_result")
            st.download_button("⬇️ Download optimized resume",
                               optimized, file_name="optimized_resume.txt",
                               mime="text/plain", key="dl_opt")

            with st.spinner("Re-scoring…"):
                new_report = ats_score(optimized, jd_text,
                                       siamese_model=siamese_model,
                                       siamese_tokenizer=siamese_tokenizer)
            delta = round(new_report["overall_score"] - score, 1)
            st.metric("🚀 ATS Score after optimization",
                      f"{new_report['overall_score']} / 100",
                      delta=f"+{delta}" if delta > 0 else str(delta))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GENERATE FROM SCRATCH (T5 GenAI)
# ─────────────────────────────────────────────────────────────────────────────
with tab_generate:
    st.subheader("✍️ Generate a Resume with AI")

    # Show which model will be used
    if _t5_is_ready(T5_GEN_CKPT):
        st.success("🤖 **T5 GenAI model** will generate your resume")
    else:
        st.info("⏳ T5 generator not trained yet — a clean template will be used.  "
                "Train it with: `python train_all.py --only-generator`")

    st.caption("Fill in your details. The AI formats everything into an ATS-friendly resume.")

    # ── Personal info ─────────────────────────────────────────────────────────
    st.markdown("#### 👤 Personal Information")
    g1, g2 = st.columns(2)
    with g1:
        gen_name     = st.text_input("Full Name",         placeholder="Alex Johnson",            key="g_name")
        gen_email    = st.text_input("Email",             placeholder="alex@email.com",           key="g_email")
        gen_phone    = st.text_input("Phone",             placeholder="+1 555 123 4567",          key="g_phone")
    with g2:
        gen_location = st.text_input("Location",          placeholder="San Francisco, CA",        key="g_loc")
        gen_linkedin = st.text_input("LinkedIn URL",      placeholder="linkedin.com/in/alex",     key="g_li")
        gen_title    = st.text_input("Desired Job Title", placeholder="Senior Python Developer",  key="g_title")

    gen_years = st.slider("Years of Experience", 0, 30, 3, key="g_years")

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown("#### 📝 Professional Summary")
    gen_summary = st.text_area(
        "Write a short professional summary (2–4 sentences)",
        placeholder="Experienced software engineer with 3+ years building scalable backend systems…",
        height=100, key="g_summary"
    )

    # ── Skills ────────────────────────────────────────────────────────────────
    st.markdown("#### 🛠️ Skills")
    gen_skills_raw = st.text_input(
        "Enter skills separated by commas",
        placeholder="Python, Docker, PostgreSQL, REST API, Git, CI/CD",
        key="g_skills"
    )

    # ── Experience ────────────────────────────────────────────────────────────
    st.markdown("#### 💼 Work Experience")
    num_exp = st.number_input("How many jobs to add?", 1, 5, 2, key="g_num_exp")
    experiences = []
    for i in range(int(num_exp)):
        with st.expander(f"Job #{i+1}", expanded=(i == 0)):
            e1, e2 = st.columns(2)
            with e1:
                exp_title   = st.text_input("Job Title",            placeholder="Software Engineer", key=f"g_et_{i}")
                exp_company = st.text_input("Company",              placeholder="Acme Corp",          key=f"g_ec_{i}")
            with e2:
                exp_start   = st.text_input("Start (e.g. Jan 2022)", placeholder="Jan 2022",         key=f"g_es_{i}")
                exp_end     = st.text_input("End (or 'Present')",    placeholder="Present",           key=f"g_ee_{i}")
            exp_bullets_raw = st.text_area(
                "Responsibilities (one per line)",
                placeholder="• Developed REST APIs using Python and Flask\n• Reduced latency by 30%",
                height=120, key=f"g_eb_{i}"
            )
            bullets = [b.lstrip("•-").strip()
                       for b in exp_bullets_raw.split("\n") if b.strip()]
            experiences.append({
                "title":   exp_title,
                "company": exp_company,
                "start":   exp_start,
                "end":     exp_end or "Present",
                "bullets": bullets,
            })

    # ── Education ─────────────────────────────────────────────────────────────
    st.markdown("#### 🎓 Education")
    num_edu = st.number_input("How many education entries?", 1, 4, 1, key="g_num_edu")
    education = []
    for i in range(int(num_edu)):
        with st.expander(f"Education #{i+1}", expanded=(i == 0)):
            d1, d2, d3 = st.columns(3)
            with d1:
                edu_degree = st.text_input("Degree",      placeholder="BSc Computer Science", key=f"g_edd_{i}")
            with d2:
                edu_inst   = st.text_input("Institution", placeholder="MIT",                  key=f"g_edi_{i}")
            with d3:
                edu_year   = st.text_input("Year",        placeholder="2021",                 key=f"g_edy_{i}")
            education.append({"degree": edu_degree, "institution": edu_inst, "year": edu_year})

    # ── Certifications ────────────────────────────────────────────────────────
    st.markdown("#### 🏅 Certifications (optional)")
    gen_certs_raw = st.text_input(
        "Enter certifications separated by commas",
        placeholder="AWS Certified Developer, Google Cloud Professional",
        key="g_certs"
    )

    # ── Target JD ─────────────────────────────────────────────────────────────
    st.markdown("#### 🎯 Target Job Description (optional but recommended)")
    gen_jd = st.text_area(
        "Paste the job description — AI will inject matching keywords and show ATS score",
        height=150, key="g_jd",
        placeholder="We are looking for a Senior Python Developer with Kubernetes experience…"
    )

    st.divider()

    # ── Generate button ───────────────────────────────────────────────────────
    if st.button("🤖 Generate Resume with AI", type="primary",
                 use_container_width=True, key="btn_generate"):

        if not gen_name.strip():
            st.error("Please enter your full name.")
        elif not gen_title.strip():
            st.error("Please enter a desired job title.")
        else:
            skills_list = [s.strip() for s in gen_skills_raw.split(",") if s.strip()]
            certs_list  = [c.strip() for c in gen_certs_raw.split(",")  if c.strip()]

            with st.spinner("🤖 AI is generating your resume…"):
                generated, used_ai = generator.generate(
                    full_name      = gen_name,
                    job_title      = gen_title,
                    email          = gen_email,
                    phone          = gen_phone,
                    location       = gen_location,
                    linkedin       = gen_linkedin,
                    years_exp      = gen_years,
                    summary        = gen_summary,
                    skills         = skills_list,
                    experiences    = experiences,
                    education      = education,
                    certifications = certs_list,
                    jd_text        = gen_jd,
                )

            # Show which model was used
            if used_ai:
                st.success("✅ Generated by T5 GenAI model")
            else:
                st.info("📋 Generated using structured template (T5 not trained yet)")

            st.subheader("📄 Your Generated Resume")
            st.text_area("Generated resume (copy & edit):",
                         generated, height=500, key="gen_result")
            st.download_button(
                "⬇️ Download as .txt",
                generated,
                file_name=f"{gen_name.replace(' ','_')}_resume.txt",
                mime="text/plain", key="dl_gen"
            )

            # ── ATS Score ─────────────────────────────────────────────────────
            if gen_jd.strip():
                st.divider()
                st.subheader("📊 ATS Score of Generated Resume")
                with st.spinner("Scoring…"):
                    gen_report = ats_score(generated, gen_jd,
                                           siamese_model=siamese_model,
                                           siamese_tokenizer=siamese_tokenizer)
                score = gen_report["overall_score"]
                color = "🟢" if score >= 70 else "🟡" if score >= 45 else "🔴"
                st.metric(f"{color} ATS Score", f"{score} / 100")
                st.progress(int(score))
                kw = gen_report["keyword_coverage"]
                with st.expander(f"🔑 Keyword Coverage  {kw['coverage_pct']}%"):
                    st.write("**Matched:**", ", ".join(kw["matched"]) or "—")
                    st.write("**Missing:**", ", ".join(kw["missing"]) or "—")
            else:
                st.info("💡 Paste a job description above to see your ATS score!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Resume2Vec  •  Siamese Transformer + T5 GenAI  •  ATS Optimization & Generation")
