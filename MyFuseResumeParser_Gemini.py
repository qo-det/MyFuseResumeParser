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
# Replace with your actual Gemini API key
GEMINI_API_KEY = "GEMINI_API_KEY"
palm.configure(api_key=GEMINI_API_KEY)

# Using a hypothetical Gemini model name
modelpalm = palm.GenerativeModel('gemini-1.5-flash')

# --------------------------
# Helper Functions
# --------------------------
def convert_file_to_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        pages = convert_from_path(file_path, dpi=150)  # PDF -> images
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
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        # OCR directly from image
        return pytesseract.image_to_string(Image.open(file_path))
    else:
        raise ValueError("Unsupported file format for text conversion: " + ext)

def gemini_generate_text_from_file(prompt, file_path, temperature=0.0):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        # Image input
        try:
            img = Image.open(file_path)
        except Exception as e:
            raise Exception("Error opening image file: " + str(e))
        full_input = [prompt, img]
        response = modelpalm.generate_content(full_input)
    elif ext in ['.txt', '.pdf', '.docx']:
        # Text input
        text = convert_file_to_text(file_path)
        full_prompt = prompt + "\n" + text
        response = modelpalm.generate_content(full_prompt)
    else:
        raise ValueError("Unsupported file format: " + ext)

    if hasattr(response, "text"):
        return response.text
    elif isinstance(response, dict) and "text" in response:
        return response["text"]
    else:
        return response.text

def remove_code_fences(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()

def extract_section(prompt, file_path, temperature=0.0):
    output = gemini_generate_text_from_file(prompt, file_path, temperature=temperature)
    return remove_code_fences(output)

def parse_json_output(output):
    try:
        return json.loads(output)
    except Exception:
        return output  # Return raw string if JSON parse fails

# --------------------------
# Example Prompts
# --------------------------
prompt_profile = """
Extract the profile details from profile section or corresponding section as JSON:
{
  "name": "[Candidate Name]",
  "position": "[Position]",
  "email": "[Email]",
  "phone": "[Phone]",
  "location": "[Location]",
  "linkedinUsername": "[LinkedIn URL]",
  "address": "[Address]",
  "tags": "[Tag1|Tag2|...]"
}
Return valid JSON only.
Text:
"""

prompt_summary = """
Extract the professional summary from summary section or corresponding section in JSON:
{
  "summary": "<p>Professional summary in HTML</p>"
}
Return valid JSON only.
Text:
"""

prompt_techskills = """
Extract skills from skills section or corresponding section in JSON:
{
  "simpleList": ["skill1","skill2","..."],
  "twoColumn": [
    {
      "name": "Technical Skills",
      "skills": ["tech1","tech2","..."]
    },
    {
      "name": "Soft Skills",
      "skills": ["soft1","soft2","..."]
    }
  ],
  "detailed": [
    {
      "name": "[Category Name]",
      "skills": ["skill1","skill2","..."]
    }
  ]
}
Return valid JSON only.
Text:
"""

prompt_experience = """
Extract experience from experience section or corresponding section as an array of objects:
[
  {
    "company": "[Company Name]",
    "role": "[Role Title]",
    "timePeriod": "[Duration]",
    "location": "[Location]",
    "points": [
      "Bullet point 1",
      "Bullet point 2"
    ]
  }
]
Return valid JSON only.
Text:
"""

prompt_education = """
Extract education from education section or corresponding section as an array of objects:
[
  {
    "degree": "[Degree Name]",
    "branch": "[Branch]",
    "college": "[College]",
    "year": "[Duration]",
    "score": "[Score/CGPA if any]"
  }
]
Return valid JSON only.
Text:
"""

prompt_projects = """
Extract projects from projects section or corresponding section as an array of objects:
[
  {
    "title": "[Project Title]",
    "organisation": "[Organisation]",
    "year": "[Project Duration]",
    "description": "<p>Summary in HTML</p>",
    "points": [
      "Bullet point 1",
      "Bullet point 2"
    ]
  }
]
Return valid JSON only.
Text:
"""

# For completeness, add empty prompts for internship, awards, achievements, etc. if needed
prompt_internship = """
Extract internship details from experience section or corresponding section as an array of objects:
[
  {
    "company": "[Internship Company]",
    "role": "[Role]",
    "timePeriod": "[Duration]",
    "points": [
      "Bullet 1",
      "Bullet 2"
    ]
  }
]
Return valid JSON only.
Text:
"""

prompt_awards = """
Extract awards from the award section or corresponding section as an array of objects:
[
  {
    "name": "[Award Name]",
    "IssuedOn": "[Date]",
    "IssuedBy": "[Issuer]"
  }
]
Return valid JSON only.
Text:
"""

prompt_achievements = """
Extract achievements from the achievements section strictly or anything that corresponds to achievements, not awards or anything else, as an array of objects:
[
  {
    "points": [
      "Achievement 1",
      "Achievement 2"
    ]
  }
]
Return valid JSON only.
Text:
"""

prompt_membership = """
Extract membership details from any section corresponding to memberships as an array of objects:
[
  {
    "organization": "[Membership Organization]",
    "role": "[Role/Title]"
  }
]
Return valid JSON only.
Text:
"""

prompt_keystrength = """
Extract key strengths as an array of strings:
["Strength 1", "Strength 2"]
Return valid JSON only.
Text:
"""

prompt_optional = """
Extract optional details not covered in other sections, as an array of objects:
[
  {
    "name": "[Section Title]",
    "points": [
      "Bullet 1",
      "Bullet 2"
    ]
  }
]
Return valid JSON only.
Text:
"""
def safe_parse(data):
    """
    Ensure that data is parsed into a dict or list.
    If data is a string, try to load it as JSON.
    Otherwise, return an empty dict (or list) as appropriate.
    """
    if isinstance(data, str):
        try:
            return json.loads(data)
        except Exception as e:
            # If parsing fails, log the error and return an empty dict.
            print(f"Error parsing data: {e}")
            return {}
    return data
# --------------------------
# Main Code: Final JSON Assembly
# --------------------------
def main():
    file_path = "/content/WhatsApp Image 2025-03-28 at 17.08.30.jpeg"
    if not file_path or not os.path.exists(file_path):
        print("File not found. Please provide a valid file path.")
        sys.exit(1)

    # 1) Extract raw responses from each prompt
    profile_raw = extract_section(prompt_profile, file_path)
    summary_raw = extract_section(prompt_summary, file_path)
    techskills_raw = extract_section(prompt_techskills, file_path)
    experience_raw = extract_section(prompt_experience, file_path)
    education_raw = extract_section(prompt_education, file_path)
    projects_raw = extract_section(prompt_projects, file_path)
    internship_raw = extract_section(prompt_internship, file_path)
    awards_raw = extract_section(prompt_awards, file_path)
    achievements_raw = extract_section(prompt_achievements, file_path)
    membership_raw = extract_section(prompt_membership, file_path)
    keystrength_raw = extract_section(prompt_keystrength, file_path)
    optional_raw = extract_section(prompt_optional, file_path)
    # 2) Parse each into Python structures using safe_parse
    profile_data = safe_parse(profile_raw)
    summary_data = safe_parse(summary_raw)
    techskills_data = safe_parse(techskills_raw)
    experience_data = safe_parse(experience_raw)
    education_data = safe_parse(education_raw)
    projects_data = safe_parse(projects_raw)
    internship_data = safe_parse(internship_raw)
    awards_data = safe_parse(awards_raw)
    achievements_data = safe_parse(achievements_raw)
    membership_data = safe_parse(membership_raw)
    keystrength_data = safe_parse(keystrength_raw)
    optional_data = safe_parse(optional_raw)

    # 3) Build the final JSON with the EXACT structure you want
    final_json = {
        "name": "Untitled",
        "sections": {
            "profile": {
                "name": "Profile",
                "key": "sections.profile",
                "data": {
                    "name": profile_data.get("name", ""),
                    "position": profile_data.get("position", ""),
                    "email": profile_data.get("email", ""),
                    "phone": profile_data.get("phone", ""),
                    "location": profile_data.get("location", ""),
                    "linkedinUsername": profile_data.get("linkedinUsername", ""),
                    "address": profile_data.get("address", ""),
                    "tags": profile_data.get("tags", "")
                }
            },
            "Professionalsummary": {
                "name": "Professional Summary",
                "key": "sections.Professionalsummary",
                "data": {
                    "summary": summary_data.get("summary", "") if isinstance(summary_data, dict) else ""
                }
            },
            "technicalSkills": {
                "name": "Technical Skills",
                "key": "sections.technicalSkills",
                "data": [],
                "activeTab": 0,
                "simpleList": techskills_data.get("simpleList", []) if isinstance(techskills_data, dict) else [],
                "twoColumn": techskills_data.get("twoColumn", []) if isinstance(techskills_data, dict) else [],
                "detailed": techskills_data.get("detailed", []) if isinstance(techskills_data, dict) else []
            },
            "experience": {
                "name": "Experience",
                "key": "sections.experience",
                "data": experience_data if isinstance(experience_data, list) else [],
                "simpleList": experience_data if isinstance(experience_data, list) else [],
                "activeTab": 0
            },
            "education": {
                "name": "Education",
                "key": "sections.education",
                "data": education_data if isinstance(education_data, list) else []
            },
            "projects": {
                "name": "Projects",
                "key": "sections.projects",
                "data": projects_data if isinstance(projects_data, list) else []
            },
            "internship": {
                "name": "Internship",
                "key": "sections.internship",
                "data": internship_data if isinstance(internship_data, list) else []
            },
            "awards": {
                "name": "Awards",
                "key": "sections.awards",
                "data": awards_data if isinstance(awards_data, list) else []
            },
            "achivements": {
                "name": "Achivements",
                "key": "sections.achivements",
                "data": achievements_data if isinstance(achievements_data, list) else []
            },
            "careerHighlight": {
                "name": "Career Highlights",
                "key": "sections.careerHighlight",
                "data": []
            },
            "membership": {
                "name": "Membership",
                "key": "sections.membership",
                "data": membership_data if isinstance(membership_data, list) else []
            },
            "keyStrength": {
                "name": "Key Strength",
                "key": "sections.keyStrength",
                "data": keystrength_data if isinstance(keystrength_data, list) else []
            },
            "optional": {
                "name": "Optional",
                "key": "sections.optional",
                "data": optional_data if isinstance(optional_data, list) else []
            }
        }
    }

    # 4) Output the final JSON
    print(json.dumps(final_json, indent=2))

if __name__ == "__main__":
    main()
