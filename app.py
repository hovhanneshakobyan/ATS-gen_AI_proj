from __future__ import annotations

import os
import tempfile
import streamlit as st

from src.diff_utils import make_unified_diff
from src.optimizer import ResumeOptimizer
from src.parser import ResumeParser
from src.scorer import ATSScorer
from src.job_scraper import scrape_job_description


# ==============================
# INIT SESSION STATE
# ==============================
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""


st.set_page_config(page_title="ATS Resume Optimizer", layout="wide")

st.title("ATS Resume Optimizer")
st.caption("Fine-tuned semantic matcher + ATS scoring + safe local suggestions")


parser = ResumeParser()
scorer = ATSScorer()
optimizer = ResumeOptimizer()


# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("Input")

    uploaded_resume = st.file_uploader(
        "Upload resume",
        type=["pdf", "docx", "txt"]
    )

    st.subheader("Job Description")

    job_url = st.text_input("Paste Job URL (optional)")

    if st.button("Scrape Job Description"):
        if job_url.strip():
            scraped_text = scrape_job_description(job_url)
            st.session_state.jd_text = scraped_text
            st.success("Job description scraped successfully!")
        else:
            st.warning("Please enter a valid URL")

    jd_text = st.text_area(
        "Or paste/edit job description",
        value=st.session_state.jd_text,
        height=350
    )

    run_btn = st.button("Analyze Resume")


# ==============================
# RESUME PARSER
# ==============================
def parse_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            return parser.parse(tmp_path, "pdf")
        if suffix == ".docx":
            return parser.parse(tmp_path, "docx")
        return parser.parse(
            uploaded_file.getvalue().decode("utf-8", errors="ignore"),
            "text"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ==============================
# MAIN PIPELINE
# ==============================
if run_btn:

    if uploaded_resume is None:
        st.error("Please upload a resume file.")
        st.stop()

    if not jd_text.strip():
        st.error("Please provide or scrape a job description.")
        st.stop()

    # Parse resume
    resume_text = parse_uploaded_file(uploaded_resume)

    # ATS scoring (before)
    with st.spinner("Scoring original resume..."):
        before = scorer.score(resume_text, jd_text)

    # Optimize resume
    with st.spinner("Generating optimized resume..."):
        opt = optimizer.optimize(resume_text, jd_text)
        optimized_resume = opt["optimized_resume"]

    # ATS scoring (after)
    with st.spinner("Scoring optimized resume..."):
        after = scorer.score(optimized_resume, jd_text)


    # ==============================
    # RESULTS UI
    # ==============================
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Original ATS Score")
        st.metric("Score", before["overall_score"])

    with c2:
        st.subheader("Optimized ATS Score")
        st.metric(
            "Score",
            after["overall_score"],
            delta=after["overall_score"] - before["overall_score"]
        )

    st.subheader("Breakdown")

    cols = st.columns(len(before["breakdown"]))
    for i, (k, v) in enumerate(before["breakdown"].items()):
        cols[i].metric(f"Before {k.title()}", v)

    cols2 = st.columns(len(after["breakdown"]))
    for i, (k, v) in enumerate(after["breakdown"].items()):
        cols2[i].metric(f"After {k.title()}", v)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Matched / Missing",
        "Semantic Analysis",
        "Suggestions",
        "Resume Text",
        "Diff",
    ])

    with tab1:
        st.markdown("### Matched Skills")
        st.write(before["keyword_report"]["matched"])

        st.markdown("### Missing Skills")
        st.write(before["keyword_report"]["missing"])

        st.markdown("### Format Warnings")
        st.write(before["format_warnings"] or ["No major format issues."])

    with tab2:
        st.markdown("### Semantic Report")
        st.json(before["semantic_report"])

    with tab3:
        st.markdown("### Suggestions")

        if not opt["suggestions"]:
            st.info("No suggestions generated.")
        else:
            for s in opt["suggestions"]:
                st.write(f"- {s}")

    with tab4:
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("### Original Resume")
            st.text_area("Original", resume_text, height=500)

        with c4:
            st.markdown("### Optimized Resume")
            st.text_area("Optimized", optimized_resume, height=500)

    with tab5:
        diff_text = make_unified_diff(resume_text, optimized_resume)

        if not diff_text.strip():
            st.info("No differences detected.")
        else:
            st.code(diff_text, language="diff")