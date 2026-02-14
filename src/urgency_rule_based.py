def detect_urgency_rule(text):
    text = text.lower()

    high_keywords = [
        "urgent", "asap", "immediately",
        "not working", "deadline", "critical",
        "important", "right now", "emergency",
        "action required", "response needed",
        "cannot access", "system down"
    ]

    medium_keywords = [
        "soon", "priority", "update required",
        "please respond", "kindly respond",
        "at the earliest", "when possible",
        "reminder", "follow up", "fyi",
        "for your review", "request",
        "schedule", "meeting", "approval",
        "confirm", "pending", "clarification",
        "feedback", "information needed",
        "please check", "looking forward"
    ]

    for word in high_keywords:
        if word in text:
            return "High"

    for word in medium_keywords:
        if word in text:
            return "Medium"

    return "Low"
