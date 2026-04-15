import requests
from bs4 import BeautifulSoup
import re

def clean_text(t):
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def scrape_job_description(url: str) -> str:
    """
    Lightweight scraper for LinkedIn, Indeed, Glassdoor, etc.
    Works on Windows + Streamlit (no Playwright).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return f"Failed to load page: {r.status_code}"

        soup = BeautifulSoup(r.text, "html.parser")

        # LinkedIn job description
        jd = soup.find("div", {"class": "show-more-less-html__markup"})
        if jd:
            return clean_text(jd.get_text())

        # Indeed job description
        jd = soup.find("div", {"id": "jobDescriptionText"})
        if jd:
            return clean_text(jd.get_text())

        # Generic paragraph fallback
        paragraphs = soup.find_all("p")
        if paragraphs:
            return clean_text(" ".join([p.get_text() for p in paragraphs]))

        return "Could not extract job description."

    except Exception as e:
        return f"Error scraping: {str(e)}"