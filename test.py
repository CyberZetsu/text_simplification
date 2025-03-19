import textstat

text = "The ontological implications of epistemological relativism, when considered through a prism of hermeneutic phenomenology..."
score = textstat.flesch_reading_ease(text)
grade_level = textstat.flesch_kincaid_grade(text)

print(f"Flesch Reading Ease: {score}")
print(f"Flesch-Kincaid Grade Level: {grade_level}")
