import textstat

text = "The ontological implications of epistemological relativism, when considered through a prism of hermeneutic phenomenology, reveal a paradoxical tension between the subjectivity of human cognition and the purported objectivity of universal truths, suggesting that the dichotomy between empirical realism and idealist abstraction might be more malleable than traditionally assumed. As such, one must critically engage with the semiotic structures embedded within cultural paradigms, recognizing that language, as a constitutive force, both shapes and constrains the conceptualization of reality, potentially rendering any attempt at unmediated, objective understanding not only elusive but inherently flawed."
score = textstat.flesch_reading_ease(text)
grade_level = textstat.flesch_kincaid_grade(text)

print(f"Flesch Reading Ease: {score}")
print(f"Flesch-Kincaid Grade Level: {grade_level}")
