import sys
import os
import json
import re
import google.generativeai as palm
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import docx

# --------------------------
# Configuration & Setup
# --------------------------
# Our API key variable is named foreverap.
foreverap = "API_KEY_HERE"  # Replace with your actual API key
palm.configure(api_key=foreverap)
# Using Gemini Flash model (gemini-1.5-flash)
modelpalm = palm.GenerativeModel('gemini-1.5-flash')

# --------------------------
# Helper Functions
# --------------------------
def convert_file_to_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        pages = convert_from_path(file_path, dpi=150)  # Reduced DPI for speed
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
        return text
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format for text conversion: " + ext)

def gemini_generate_text_from_file(prompt, file_path, temperature=0.0):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        try:
            img = Image.open(file_path)
        except Exception as e:
            raise Exception("Error opening image file: " + str(e))
        full_input = [prompt, img]
        response = modelpalm.generate_content(full_input)
    elif ext in ['.txt', '.pdf', '.docx']:
        text = convert_file_to_text(file_path)
        full_prompt = prompt + "\n" + text
        response = modelpalm.generate_content(full_prompt)
    else:
        raise ValueError("Unsupported file format: " + ext)
    if hasattr(response, "result"):
        return response.result
    elif isinstance(response, dict) and "result" in response:
        return response["result"]
    else:
        return response

def remove_code_fences(text):
    if not isinstance(text, str):
        if hasattr(text, "text"):
            text = text.text
        else:
            text = str(text)
    # Remove any leading/trailing markdown fences (like ```json or ```).
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()

def extract_section(prompt, file_path, temperature=0.0):
    output = gemini_generate_text_from_file(prompt, file_path, temperature=temperature)
    return remove_code_fences(output)

def parse_output(output):
    try:
        return json.loads(output)
    except Exception:
        return output

def tidy_optional(optional_output):
    """
    Ensures that the optional section is a JSON array of objects
    with keys "name" and "description" (no extra escapes or newline characters).
    """
    try:
        data = json.loads(optional_output) if isinstance(optional_output, str) else optional_output
        if isinstance(data, list):
            cleaned = []
            for entry in data:
                if isinstance(entry, dict):
                    name = entry.get("name", "").strip()
                    description = entry.get("description", "").strip()
                    if name and description:
                        cleaned.append({"name": name, "description": description})
            return cleaned
        else:
            return data
    except Exception:
        return optional_output

# --------------------------
# Prompts for Each Section
# --------------------------
prompt_profile = """
Extract the profile details from the text below. Return the information in the following JSON format exactly:
{
  "name": "Your Name",
  "phone": "Phone",
  "mail": "yourmail@gamil.com",
  "github": "Github Url",
  "linkedin": "LinkedIn Url"
}
Return only valid JSON without extra formatting.
Text:
"""

prompt_education = """
Extract the education details from the text below. For each education entry, return the following information exactly in a JSON array:
- degree: Degree Name
- university: University Name
- college: College Name (if applicable, otherwise null)
- branch: Branch or specialisation (if applicable, otherwise null)
- year: Duration of study (e.g. 2021-2025)
- cgpa: CGPA (if available)
- percentage: Percentage (if available)
Return only valid JSON without any extra markdown formatting.
Text:
"""

prompt_experience = """
Extract the work experience details from the text below. For each work experience entry, return the following information exactly in a JSON array:
- points: an array of bullet points summarizing the responsibilities.
- company: exp Company Name.
- link: Link (if available, otherwise empty string).
- role: Role title.
- location: Location (if available, otherwise empty string).
- timePeriod: Duration of employment.
- type: Employment type (if available, otherwise empty string).
- techstack: Technologies used (if available, otherwise empty string).
- description: A summary of responsibilities, wrapped in HTML paragraph tags.
Return only valid JSON.
Text:
"""

prompt_projects = """
Extract the project details from the text below. For each project entry, return the following information exactly in a JSON array:
- points: an array of bullet points summarizing the project.
- title: Project Title.
- organisation: Organisation or client name (if available, otherwise empty string).
- year: Project duration or year (if available, otherwise empty string).
- techstack: Technologies used (as a string, if available, otherwise empty string).
- description: A summary of the project, wrapped in HTML paragraph tags.
Return only valid JSON.
Text:
"""

prompt_techskills = """
Extract the technical skills from the text below. Return the information exactly as a JSON array where each element has:
- name: Category name (e.g., Languages, Frameworks, Tools).
- skills: an array of individual skills.
Return only valid JSON.
Text:
"""

prompt_profsummary = """
Extract the professional summary from the text below. Return the information as JSON with a single key "summary" whose value is the summary wrapped in HTML paragraph tags.
Return only valid JSON.
Text:
"""

prompt_awards = """
Extract the awards from the text below. When you encounter a section specifically for awards, extract only those entries that are clearly awards (not certificates or degrees). For each award entry, return the following information exactly in a JSON array:
- name: Award or certification name.
- description: A description wrapped in HTML paragraph tags.
Return only valid JSON.
Text:
"""

prompt_certifications = """
Extract the certifications from the text below. For each certification entry, return the following information exactly in a JSON array:
- name: Certification name.
- description: A description wrapped in HTML paragraph tags.
Return only valid JSON.
Text:
"""

prompt_optional = """
Extract the additional (optional) details from the text below, but do not include any information that has been extracted in the Profile, Education, Experience, Projects, Technical Skills, Professional Summary, Awards, or Certifications sections. For each remaining optional entry, return the following information exactly in a JSON array:
- name: Title of the optional information.
- description: Description wrapped in HTML paragraph tags.
Return only valid JSON.
Text:
"""

# --------------------------
# Main Code: Final JSON Assembly
# --------------------------
def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not file_path or not os.path.exists(file_path):
        print("File not found. Please provide a valid file path as a command-line argument.")
        sys.exit(1)

    # Strictly use file content for every section.
    profile_output = extract_section(prompt_profile, file_path)
    education_output = extract_section(prompt_education, file_path)
    experience_output = extract_section(prompt_experience, file_path)
    projects_output = extract_section(prompt_projects, file_path)
    techskills_output = extract_section(prompt_techskills, file_path)
    profsummary_output = extract_section(prompt_profsummary, file_path)
    awards_output = extract_section(prompt_awards, file_path)
    certifications_output = extract_section(prompt_certifications, file_path)
    optional_output = extract_section(prompt_optional, file_path)

    final_json = {
        "name": "Untitled",
        "sections": {
            "profile": {
                "name": "Profile",
                "key": "sections.profile",
                "data": parse_output(profile_output)
            },
            "education": {
                "name": "Education",
                "key": "sections.education",
                "data": parse_output(education_output)
            },
            "experience": {
                "name": "Experience",
                "key": "sections.experience",
                "data": parse_output(experience_output)
            },
            "projects": {
                "name": "Projects",
                "key": "sections.projects",
                "data": parse_output(projects_output)
            },
            "technicalSkills": {
                "name": "Technical Skills",
                "key": "sections.technicalSkills",
                "data": parse_output(techskills_output)
            },
            "Professionalsummary": {
                "name": "Professional Summary",
                "key": "sections.Professionalsummary",
                "data": parse_output(profsummary_output)
            },
            "awards": {
                "name": "Awards",
                "key": "sections.awards",
                "data": parse_output(awards_output)
            },
            "certifications": {
                "name": "Certifications",
                "key": "sections.certifications",
                "data": parse_output(certifications_output)
            },
            "optional": {
                "name": "Optional",
                "key": "sections.optional",
                "data": tidy_optional(parse_output(optional_output))
            }
        }
    }

    
    print(json.dumps(final_json, indent=2))

if __name__ == "__main__":
    main()
