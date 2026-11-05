from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data: example notices and their labels
TRAINING_DATA = [
    # exams
    ("End of semester exams start on 2 December in the main hall", "exams"),
    ("CAT 2 for BIT 101 will be held on Monday at 8am", "exams"),
    ("Exam timetable is now available on the student portal", "exams"),
    ("Supplementary exams registration closes next week", "exams"),
    ("Students must carry their exam cards to every paper", "exams"),
    ("Special exam applications should be submitted to the dean", "exams"),
    ("Results for last semester are out on the portal", "exams"),
    ("Assignment submission is due on Friday", "exams"),
    ("Re-sit examinations will be held in January", "exams"),
    ("Exam rooms allocation has been posted on the notice board", "exams"),
    # fees
    ("Fee balance must be cleared before 25th November", "fees"),
    ("Pay tuition through the university paybill before the deadline", "fees"),
    ("Students with outstanding fees will not sit exams", "fees"),
    ("Fee structure for the new semester has been published", "fees"),
    ("HELB loan forms are available at the finance office", "fees"),
    ("Late payment of fees attracts a penalty", "fees"),
    ("Fee clearance is required before collecting your results", "fees"),
    ("Instalment payment plans are available at the finance office", "fees"),
    ("Download your fee statement from the student portal", "fees"),
    ("Bursary applications close on 30th November", "fees"),
    # events
    ("Join the tech club hackathon this Friday at the ICT lab", "events"),
    ("Career fair happening in the main auditorium on Thursday", "events"),
    ("Inter-faculty football tournament kicks off this weekend", "events"),
    ("Come for the club open day and meet new members", "events"),
    ("Guest lecture on artificial intelligence by an industry speaker", "events"),
    ("Graduation rehearsal will be held at the main field", "events"),
    ("Cultural day celebrations start at 10am in the quadrangle", "events"),
    ("Debate club invites all students to the inter-university debate", "events"),
    ("Charity walk organised by the student council on Saturday", "events"),
    ("Workshop on CV writing and interview skills this Wednesday", "events"),
    # general
    ("The library will be closed on Saturday for maintenance", "general"),
    ("Water will be unavailable in the hostels tomorrow morning", "general"),
    ("Student ID replacement is now done at the registrar's office", "general"),
    ("Wifi upgrade in progress, expect interruptions in the labs", "general"),
    ("Hostel booking for next semester opens at the accommodation office", "general"),
    ("The cafeteria has new opening hours", "general"),
    ("Report lost items to the security office", "general"),
    ("Parking stickers are issued at the administration block", "general"),
    ("Power outage expected in the science block this afternoon", "general"),
    ("Clinic services are available from 8am to 5pm", "general"),
]

_texts = [text for text, _ in TRAINING_DATA]
_labels = [label for _, label in TRAINING_DATA]

_vectorizer = TfidfVectorizer(stop_words="english")
_model = LogisticRegression(C=10, max_iter=1000)
_model.fit(_vectorizer.fit_transform(_texts), _labels)


def classify_notice(text):
    """Return (category, confidence between 0 and 1)."""
    probabilities = _model.predict_proba(_vectorizer.transform([text]))[0]
    best = probabilities.argmax()
    return _model.classes_[best], float(probabilities[best])


if __name__ == "__main__":
    tests = [
        ("Clear your tuition balance by 30th November", "fees"),
        ("Hackathon this Friday, register now", "events"),
        ("Final exams timetable released", "exams"),
        ("The main gate will be closed for repairs", "general"),
        ("Pay your fees via paybill", "fees"),
        ("Guest speaker on machine learning next week", "events"),
        ("CAT 1 results are out", "exams"),
        ("Water shortage in the hostels", "general"),
    ]
    correct = 0
    for text, expected in tests:
        label, confidence = classify_notice(text)
        correct += label == expected
        print(f"{label:8} {confidence:.2f}  (expected {expected:8}) <- {text}")
    print(f"{correct}/{len(tests)} correct")