from datetime import date, datetime

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from classifier import classify_notice
from extract_dates import extract_dates

st.set_page_config(page_title="Campus Notice Assistant", page_icon="🎓")

CATEGORY_ICONS = {"exams": "📝", "fees": "💰", "events": "🎉", "general": "📢"}
LOW_CONFIDENCE = 0.40  # below this, warn the user the category may be wrong

SAMPLE_NOTICES = [
    "Fee balance must be cleared before 25th November",
    "End of semester exams start on 2 December in the main hall",
    "Join the tech club hackathon this Friday at the ICT lab",
    "The library will be closed on Saturday for maintenance",
    "CAT 2 for BIT 101 will be held on 12/11/2026 at 8am",
    "Bursary applications close on 30th November",
]


# ---------- helpers ----------

def analyse(text):
    """Run the AI pieces on one notice: category + dates."""
    category, confidence = classify_notice(text)
    return {
        "text": text,
        "category": category,
        "confidence": confidence,
        "dates": extract_dates(text, datetime.now()),
    }


def find_matches(question, notices, top_n=3, min_score=0.1):
    """Return the notices most similar to the question."""
    texts = [n["text"] for n in notices]
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:  # nothing but stop words
        return []
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    ranked = scores.argsort()[::-1][:top_n]
    return [(notices[i], scores[i]) for i in ranked if scores[i] >= min_score]


def make_ics(items):
    """Build a calendar file (.ics) with one all-day event per deadline."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Campus Notice Assistant//EN"]
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for i, (title, day) in enumerate(items):
        safe = title[:80].replace(",", " ").replace(";", " ")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{i}-{day:%Y%m%d}@campus-notice-assistant",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{day:%Y%m%d}",
            f"SUMMARY:{safe}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)


def show_notice(n):
    icon = CATEGORY_ICONS.get(n["category"], "📢")
    st.markdown(f"**{icon} {n['category'].title()}**  ·  confidence {n['confidence']:.0%}")
    if n["confidence"] < LOW_CONFIDENCE:
        st.warning("⚠️ Not sure about this category, please check it.")
    st.write(n["text"])
    if n["dates"]:
        for phrase, day in n["dates"]:
            st.caption(f"📅 “{phrase}” → {day:%A, %d %B %Y}")


# ---------- app state ----------

if "notices" not in st.session_state:
    st.session_state.notices = [analyse(t) for t in SAMPLE_NOTICES]

st.title("🎓 Campus Notice Assistant")
st.write("Paste campus notices, and the assistant sorts them, finds the deadlines, and answers your questions.")

tab_add, tab_ask, tab_dates = st.tabs(["➕ Add notices", "💬 Ask a question", "📅 Deadlines"])

# ---------- tab 1: add notices ----------
with tab_add:
    st.write("Paste one or more notices. Separate different notices with a blank line.")
    raw = st.text_area("Notice text", height=150, key="raw_notice")
    if st.button("Analyse and save", type="primary"):
        parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if not parts:
            st.warning("Paste a notice first.")
        else:
            for part in parts:
                result = analyse(part)
                st.session_state.notices.append(result)
                show_notice(result)
                st.divider()
            st.success(f"Saved {len(parts)} notice(s).")

    with st.expander(f"All saved notices ({len(st.session_state.notices)})"):
        for n in st.session_state.notices:
            show_notice(n)
            st.divider()

# ---------- tab 2: ask ----------
with tab_ask:
    question = st.text_input("Ask about your notices", placeholder="When is the fee deadline?", key="question")
    if question:
        matches = find_matches(question, st.session_state.notices)
        if not matches:
            st.info("Sorry, I couldn't find a notice about that.")
        else:
            best, score = matches[0]
            st.subheader("Best answer")
            show_notice(best)
            st.caption(f"Match score: {score:.2f}")
            if len(matches) > 1:
                st.subheader("Also related")
                for n, s in matches[1:]:
                    show_notice(n)
                    st.divider()

# ---------- tab 3: deadlines ----------
with tab_dates:
    today = date.today()
    upcoming = []
    for n in st.session_state.notices:
        for phrase, day in n["dates"]:
            if day >= today:
                upcoming.append((day, n["category"], n["text"]))
    upcoming.sort()

    if not upcoming:
        st.info("No upcoming dates found in your notices.")
    else:
        # Plain markdown table (st.dataframe needs pyarrow, which some Windows setups block)
        rows = ["| Date | Category | Notice |", "|---|---|---|"]
        for d, c, t in upcoming:
            rows.append(f"| {d:%a %d %b %Y} | {c.title()} | {t.replace('|', '/')} |")
        st.markdown("\n".join(rows))
        st.download_button(
            "📥 Download calendar (.ics)",
            data=make_ics([(t, d) for d, _, t in upcoming]),
            file_name="campus_deadlines.ics",
            mime="text/calendar",
        )