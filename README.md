# **MyFuse Resume Parser Documentation**

This project—**MyFuse Resume Parser**—leverages Google Gemini (via the `google-generativeai` Python SDK) to extract structured resume details into strict JSON format. The parser processes an uploaded file (PDF, DOCX, TXT, or image), extracts multiple sections (Profile, Education, Experience, Projects, Technical Skills, Professional Summary, Awards, Certifications, and Optional), and returns a unified JSON object that conforms to a specific schema. An ExpressJS server is provided that accepts file uploads, invokes the Python extraction script, and returns the final JSON result to the client.

---

## **Table of Contents**

- Overview

- Installation

- Project Structure

- Python Script: `MyFuseResumeParser.py`

  - Configuration and Setup

  - Helper Functions

  - Section Prompts and Extraction

  - Final JSON Assembly

- ExpressJS Server Integration

  - Express Code Overview

  - Integration Instructions

- Usage

- Notes

---

## **Overview**

**MyFuse Resume Parser** automates resume extraction by:

- **File Handling:** Accepting an uploaded file (e.g., PDF, DOCX, TXT, or image) and processing its contents.

- **Section Extraction:** Using section-specific prompts to instruct Gemini to return strict JSON data for each resume section.

- **Output Cleanup:** Removing extra markdown formatting (such as code fences) and converting the output into valid JSON.

- **Final Assembly:** Combining the extracted sections into one final JSON object that matches the required schema.

- **ExpressJS Integration:** Providing an endpoint that accepts file uploads and returns the parsed JSON.

---

## **Installation**

### **Python Dependencies**

Install the required Python packages (for example, in your terminal or a Google Colab cell):

bash  
Copy  
`pip install -q -U google-generativeai pytesseract pdf2image python-docx Pillow`  
`apt-get install -y poppler-utils`

### **Node.js / Express Dependencies**

In your Node.js project directory, run:

bash  
Copy  
`npm install express multer body-parser`

---

## **Project Structure**

A typical project structure might be:

bash  
Copy  
`/project-root`  
 `|-- gemini_extractor.py   # Python extraction script (MyFuse Resume Parser)`  
 `|-- server.js             # ExpressJS server integration`  
 `|-- uploads/              # Folder to store uploaded files`  
 `|-- package.json          # Node.js project file`  
 `|-- README.md             # This documentation file`

---

## **Python Script: `gemini_extractor.py`**

This script is the core of **MyFuse Resume Parser**. It:

- Configures the Gemini API using an API key stored in the variable `foreverap`.

- Processes the input file (passed as a command‑line argument) by converting it to text (or using the image directly).

- Uses section-specific prompts to extract data into strict JSON.

- Cleans up the output (removing markdown fences) and assembles the final JSON according to a predefined schema.

### **Configuration and Setup**

- **API Key:**  
   The script uses the API key from the variable `foreverap`. Replace `"YOUR_FOREVERAP_API_KEY"` with your actual key.

- **Model:**  
   It creates a Gemini GenerativeModel instance using `'gemini-1.5-flash'`.

### **Helper Functions**

- **`convert_file_to_text(file_path)`**  
   Converts PDFs (using `pdf2image` and `pytesseract`), DOCX files (using `python-docx`), or TXT files to plain text.

- **`gemini_generate_text_from_file(prompt, file_path, temperature=0.0)`**  
   Depending on the file extension, either passes an image or appends text from the file to the prompt and calls Gemini’s API.

- **`remove_code_fences(text)`**  
   Removes any extra markdown formatting (e.g., "\`\`\`json") from the output.

- **`extract_section(prompt, file_path, temperature=0.0)`**  
   Combines the prompt with file content and returns the cleaned output from Gemini.

- **`parse_output(output)`**  
   Attempts to convert the output string into a Python JSON object using `json.loads()`.

- **`tidy_optional(optional_output)`**  
   Ensures that the optional section is a JSON array of objects with keys `"name"` and `"description"` without extra escapes or newlines.

### **Section Prompts and Extraction**

Each section has its own prompt (e.g., for Profile, Education, Experience, Projects, Technical Skills, Professional Summary, Awards, Certifications, and Optional) that instructs Gemini to extract the required data strictly as JSON. The optional prompt is tailored to exclude information that was already captured in other sections.

### **Final JSON Assembly**

After extracting all sections, the script assembles them into a single JSON object with the following structure:

json  
Copy  
`{`  
 `"name": "Untitled",`  
 `"sections": {`  
 `"profile": { ... },`  
 `"education": { ... },`  
 `"experience": { ... },`  
 `"projects": { ... },`  
 `"technicalSkills": { ... },`  
 `"Professionalsummary": { ... },`  
 `"awards": { ... },`  
 `"certifications": { ... },`  
 `"optional": { ... }`  
 `}`  
`}`

The final JSON is printed to stdout.

_(The full Python code is provided in the previous section.)_

---

## **ExpressJS Server Integration**

The ExpressJS server acts as a middleware interface to call the Python extraction script and return the result.

### **Express Code Overview**

- **Multer:**  
   The server uses Multer to handle file uploads, storing files in an `uploads/` folder.

- **Endpoint `/extract`:**  
   A POST endpoint that accepts file uploads. The file path is passed as an argument to the Python script via Node’s `child_process.spawn`.

- **Output Handling:**  
   The Python script’s stdout (which is the final JSON) is captured, parsed, and returned as the HTTP response.

### **Integration Instructions**

To integrate **MyFuse Resume Parser** into an existing ExpressJS codebase:

**Add the Python Extraction Script:**  
 – Copy `gemini_extractor.py` into your project. – Ensure it works independently by running it with a file path argument:

`python gemini_extractor.py /content/Devanshu\ Choudhary-\ Resume.pdf`

1.
2.  **Update Your Express Server:**  
    – If you already have an Express server, add a new route (or integrate with an existing one) that handles file uploads using Multer. – Use Node’s `child_process.spawn` to invoke the Python script with the uploaded file’s path as a parameter.

3.  **File Upload Setup:**  
    – Ensure your Express server saves uploaded files in a folder (e.g., `uploads/`). – The server then passes the complete file path to the Python script.

4.  **Return the Final JSON:**  
    – Capture the Python script’s stdout, parse it as JSON, and send it as your Express endpoint’s response.

_(The complete Express code is provided in the next section.)_

---

## **ExpressJS Server Code (server.js)**

Below is a sample Express server code:

javascript  
Copy  
`const express = require('express');`  
`const multer  = require('multer');`  
`const bodyParser = require('body-parser');`  
`const { spawn } = require('child_process');`  
`const path = require('path');`

`const app = express();`  
`const port = 3000;`

`// Configure multer to save uploaded files to 'uploads/' folder`  
`const storage = multer.diskStorage({`  
 `destination: (req, file, cb) => {`  
 `cb(null, 'uploads/');`  
 `},`  
 `filename: (req, file, cb) => {`  
 `cb(null, file.originalname);`  
 `}`  
`});`  
`const upload = multer({ storage: storage });`

`// Middleware`  
`app.use(bodyParser.json());`  
`app.use(bodyParser.urlencoded({ extended: true }));`

`// Endpoint to upload file and extract data using the Python script`  
`app.post('/extract', upload.single('file'), (req, res) => {`  
 `if (!req.file) {`  
 `return res.status(400).send({ error: 'No file uploaded.' });`  
 `}`  
 `const filePath = path.join(__dirname, req.file.path);`

`// Spawn the Python process; pass the file path as an argument.`  
 `const pythonProcess = spawn('python', ['gemini_extractor.py', filePath]);`

`let dataToSend = '';`  
 `pythonProcess.stdout.on('data', (data) => {`  
 `dataToSend += data.toString();`  
 `});`

`pythonProcess.stderr.on('data', (data) => {`  
 `` console.error(`stderr: ${data}`); ``  
 `});`

`pythonProcess.on('close', (code) => {`  
 `if (code !== 0) {`  
 `return res.status(500).send({ error: 'Python script exited with code ' + code });`  
 `}`  
 `try {`  
 `const jsonOutput = JSON.parse(dataToSend);`  
 `res.json(jsonOutput);`  
 `} catch (err) {`  
 `res.status(500).send({ error: 'Error parsing JSON output', details: dataToSend });`  
 `}`  
 `});`  
`});`

`// Start the server`  
`app.listen(port, () => {`  
 `` console.log(`Express server running at http://localhost:${port}`); ``  
`});`

---

## **Usage**

1. **Install Dependencies:**  
   – Install Python packages and Node.js packages as described above.

2. **Configure the Python Script:**  
   – Save `gemini_extractor.py` and replace `"YOUR_FOREVERAP_API_KEY"` with your actual key.

**Run the Express Server:**  
 – Save `server.js` in your project and create an `uploads/` folder. – Run:

bash  
Copy  
`node server.js`

3.
4.  **Send a Request:**  
    – Use Postman (or your front-end) to send a POST request to `http://localhost:3000/extract` with the file field named `file`.  
    – The server will pass the file path to the Python script, which extracts all sections and returns the final JSON.

---

## **Notes**

- **Strict JSON Output:**  
   The prompts instruct Gemini to return valid JSON. The Python script cleans the output (removing extra code fences) and parses it into Python objects.

- **Optional Section:**  
   The optional prompt instructs Gemini to ignore details that are already extracted in other sections. The `tidy_optional()` function ensures that each optional entry is a proper JSON object with `"name"` and `"description"`.

- **Integration:**  
   To integrate **MyFuse Resume Parser** into an existing ExpressJS project, simply add the Python extraction script and create/update your Express route to call the Python script using `child_process.spawn`.
