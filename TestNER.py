import re
import spacy
import json

# Load the spaCy large English model
nlp = spacy.load("en_core_web_lg")

def extract_profile_info(profile_text):
    """
    Extracts profile details from the given text.
    
    - Uses spaCy to find PERSON entities (names).
    - Uses regex to extract email addresses, phone numbers, and links.
    - Removes the extracted fields from the text to form the summary.
    
    Returns a dictionary with the following keys:
      - name: The first PERSON entity detected (or empty string if none).
      - email: The first email address found.
      - phone: The first phone number found.
      - links: A list of all links found.
      - summary: The remaining text after the above fields are removed.
    """
    # Process the text with spaCy
    doc = nlp(profile_text)
    
    # Extract names (all PERSON entities); take the first one as "name"
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    name = names[0] if names else ""
    
    # Use regex patterns to extract email, phone numbers, and links
    email_matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', profile_text)
    phone_matches = re.findall(r'\+?\d[\d\-\s]{8,}\d', profile_text)
    link_matches = re.findall(r'https?://\S+', profile_text)
    
    # Create a summary by removing extracted fields from the original text.
    summary = profile_text
    for field in email_matches + phone_matches + names + link_matches:
        summary = re.sub(re.escape(field), "", summary, flags=re.IGNORECASE)
    # Clean up extra whitespace
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    profile_data = {
        "name": name,
        "email": email_matches[0] if email_matches else "",
        "phone": phone_matches[0] if phone_matches else "",
        "links": list(set(link_matches)),  # remove duplicates if any
        "summary": summary,
        "organisations": orgs
    }
    return profile_data

# Example usage:
if __name__ == "__main__":
    # Sample profile text (replace with your extracted profile section text)
    sample_profile_text = """
 - Header: WORK EXPERIENCE ---
  > Full Stack Engineer                                                        CSD Instruments Pvt. Ltd.                                                                     Jun 2024 – Present
  > A company that specialises in manufacturing hardware products for the power sector and provides electrical fault location services.
  > Developed and launched the company website using LAMP stack, enhancing lead generation through forms and integrating Mailchimp autoresponder.
  > Designed and implemented an expense tracking mobile application, reducing fraudulent billing cases by 36%, leading to an 11% cost-saving on each field job.
  > Integrated Arduino-based GPS tracking functionality for their latest product line and made a dashboard for product tracking using Google Maps API.
  > Provided my service as a team lead for software R&D for 2.5 months
  > Freelancer                                                   Self-employed                                                                        Jan 2024 – Jun 2024
  > Took time to learn new skills and tech stacks.
  > Developed a MERN-based website for Manohar Villa, a hotel in Rajasthan, integrating lead generation through forms and linked to Makemytrip.
  > Created a MERN stack job portal, developing authentication functionality for SMTrade, an online trading platform, using JWT and cookies.
  > Published Dutch, an app built with react native featuring OAuth integration, designed to help users split expenses.
  > Graduate Engineer Trainee                   Daimler Trucks Pvt. Ltd.                                                              Feb 2023 – Jan 2024 Daimler Trucks is a global leader in commercial vehicle manufacturing, specialising in producing heavy-duty trucks and buses.
  > Contributed to the development of the latest Actros truck line for DTAG, automating the testing of 70+ test cases using deep learning.
  > Developed testing software for the ECUs of trucks using Python and C++ reducing the time taken in testing by 14%.
  > Optimised problem-solving and diagnostic tools for identifying failures in test cases.
  > Provided expertise in electronics, software development, and the BTV part of truck development.
  > Automated vehicle issue detection database using Python and machine learning to predict potential failures with 98.4% accuracy.
  > Research Intern                     PRDC Pvt. Ltd.                                                                          May 2022 – Jul 2022
  > PRDC is a company that develops software solutions for electric grids.
  > Developed a PV emulator using linear regression, lookup tables, and an 8086 microcontroller to simulate solar plant operations, built at just Rs. 2108, as to standard Rs. 14,500, reducing cost by 85%.
  > Contributed to the development of a software product line for simulating electric circuits and grids using Python and C#
  > Optimised the power grid for a steel plant, reducing energy inefficiencies.
    """
    profile_info = extract_profile_info(sample_profile_text)
    print(json.dumps(profile_info, indent=2))
