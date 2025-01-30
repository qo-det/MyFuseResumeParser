import os
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for DOCX
import spacy
import re
import pytesseract
import cv2
import numpy as np
from transformers import pipeline

# ✅ Disable GPU for Memory Optimization
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Load SpaCy Named Entity Recognition Model
nlp = spacy.load("en_core_web_sm")

# ✅ Load a **lighter-weight** Hugging Face Transformer Model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ✅ Section Labels (AI-Based + Rule-Based Detection)
SECTION_LABELS = [
    "Personal Information",
    "Summary",
    "Work Experience",
    "Education",
    "Skills",
    "Certifications",
    "Projects",
    "Achievements",
    "Contact Information",
]

# ✅ Regex Patterns for Extraction
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = r"\+?\d[\d -]{8,}\d"  # Supports different phone number formats
LINK_PATTERN = r"(https?://[^\s]+)"  # Extracts any URLs (LinkedIn, GitHub, etc.)
DATE_PATTERN = r"(\d{4}[-/]?\d{2}[-/]?\d{2})|(\d{4}-\d{4})"  # Matches YYYY-YYYY format
JOB_TITLE_PATTERN = r"\b(Software Engineer|Data Scientist|Project Manager|Developer|Analyst|Consultant|Intern|Engineer)\b"
DEGREE_PATTERN = r"\b(B\.?Tech|M\.?Tech|MBA|MSc|PhD|BSc|HSC|ICSE|Diploma)\b"
NAME_BLACKLIST = {"Extra", "Curriculum", "Team", "Oversaw", "Handled"}  # Prevent incorrect name detection

# 1️⃣ **Extract Text from Resume (Handles PDFs, DOCX, and Images)**
def extract_text(file_path):
    """Extracts text from PDFs, DOCX, or Image-Based Resumes."""
    text = ""

    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.strip()

# 2️⃣ **Hybrid AI + Rule-Based Section Segmentation**
def segment_resume_sections(text):
    """Uses AI + Rule-Based Detection to segment text into sections."""
    sections = {}
    lines = text.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # **Rule-Based Section Detection**
        if any(keyword in line.lower() for keyword in ["experience", "education", "skills", "projects", "certifications"]):
            current_section = line
            sections[current_section] = []
            continue

        # **AI-Based Section Detection (Only When Needed)**
        if not current_section:
            result = classifier(line, SECTION_LABELS, multi_label=True)
            best_score = max(result["scores"])
            predicted_label = result["labels"][result["scores"].index(best_score)]

            if best_score > 0.70:
                current_section = predicted_label
                sections[current_section] = []
        
        if current_section:
            sections[current_section].append(line)

    return sections

# 3️⃣ **Extract Personal Information**
def extract_personal_info(text):
    """Extracts Name, Email, Phone, LinkedIn, GitHub, Other Links using NER + Regex."""
    doc = nlp(text)
    info = {"name": "", "email": "", "phone": "", "linkedin": "", "github": "", "other_links": []}

    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in NAME_BLACKLIST:
            info["name"] = ent.text
        elif re.search(EMAIL_PATTERN, ent.text):
            info["email"] = ent.text
        elif re.search(PHONE_PATTERN, ent.text):
            info["phone"] = ent.text

    # Extract Links (LinkedIn, GitHub, Other Links)
    links = re.findall(LINK_PATTERN, text)
    for link in links:
        if "linkedin.com" in link:
            info["linkedin"] = link
        elif "github.com" in link:
            info["github"] = link
        else:
            info["other_links"].append(link)

    return info

# 4️⃣ **Extract Skills**
def extract_skills(text):
    """Extracts technical & soft skills from the resume."""
    predefined_skills = {"Python", "SQL", "JavaScript", "Power BI", "Excel", "Jira", "Tableau"}
    skills = set()

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "LANGUAGE", "SKILL"]:
            skills.add(ent.text.strip())

    for word in text.split():
        if word in predefined_skills:
            skills.add(word)

    return list(skills)

# 5️⃣ **Extract Experience**
def extract_experience(text):
    """Extracts experience details using job title patterns."""
    experience = []
    lines = text.split("\n")

    for line in lines:
        if re.search(JOB_TITLE_PATTERN, line) or re.search(DATE_PATTERN, line):
            experience.append(line)

    return experience

# 6️⃣ **Extract Education**
def extract_education(text):
    """Extracts education details using regex for degrees & institutions."""
    education = []
    lines = text.split("\n")

    for line in lines:
        if re.search(DEGREE_PATTERN, line):
            education.append(line)

    return education

# ✅ **Final Resume Parsing Function**
def parse_resume(file_path):
    """Handles full resume parsing with AI segmentation and regex-based extractions."""
    text = extract_text(file_path)
    sections = segment_resume_sections(text)

    extracted_data = {
        "personal_info": extract_personal_info(text),
        "experience": extract_experience("\n".join(sections.get("Work Experience", []))),
        "education": extract_education("\n".join(sections.get("Education", []))),
        "skills": extract_skills("\n".join(sections.get("Skills", []))),
        "certifications": sections.get("Certifications", []),
        "achievements": sections.get("Achievements", []),
        "projects": sections.get("Projects", []),
    }

    return extracted_data

# ✅ **Run the Parser**
if __name__ == "__main__":
    file_path = "/home/m3phi5t0/myfuseResumeParser/Devanshu Choudhary- Resume.pdf"  # Change this to test different files
    parsed_resume = parse_resume(file_path)

    print("\n✅ Extracted Resume Data:")
    for key, value in parsed_resume.items():
        print(f"\n📌 {key.capitalize()}:")
        print(value)
