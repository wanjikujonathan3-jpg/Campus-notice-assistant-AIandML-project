import re
from datetime import datetime
import dateparser

MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
WEEKDAYS = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?"

PATTERNS = [
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTHS}\b(?:\s+\d{{4}})?",   # 25th November, 2 Dec 2026
    rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}})?",           # November 25
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",                                     # 25/11/2026
    rf"\b(?:this|next)\s+{WEEKDAYS}\b",                                       # this Friday
    rf"\bon\s+{WEEKDAYS}\b",                                                  # on Saturday
    r"\b(?:today|tomorrow)\b",
    r"\bnext\s+week\b",
]


def extract_dates(text, today=None):
    today = today or datetime.now()
    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": today,
        "DATE_ORDER": "DMY",  # Kenya writes day first
    }
    results = []
    for pattern in PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = match.group(0)
            phrase_clean = re.sub(r"^on\s+", "", phrase, flags=re.IGNORECASE)
            parsed = dateparser.parse(phrase_clean, settings=settings)
            if parsed:
                results.append((phrase, parsed.date()))
    return results


if __name__ == "__main__":
    test_today = datetime(2026, 11, 5)  # fixed "today" so results are repeatable
    notices = [
        "Fee balance must be cleared before 25th November",
        "End of semester exams start on 2 December in the main hall",
        "Join the tech club hackathon this Friday at the ICT lab",
        "The library will be closed on Saturday for maintenance",
        "CAT 2 will be held on 12/11/2026 at 8am",
        "Water will be unavailable tomorrow morning",
        "Welcome back to campus, enjoy the semester",
    ]
    for text in notices:
        print(text)
        print("   ->", extract_dates(text, test_today))