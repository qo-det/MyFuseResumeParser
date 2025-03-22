const express = require("express");
const multer = require("multer");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
const port = 3000;

// Configure multer to save uploaded files to 'uploads/' folder
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  },
});
const upload = multer({ storage: storage });

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Endpoint to upload file and extract data using the Python script
app.post("/extract", upload.single("file"), (req, res) => {
  if (!req.file) {
    return res.status(400).send({ error: "No file uploaded." });
  }
  const filePath = path.join(__dirname, req.file.path);

  // Spawn the Python process; pass the file path as an argument.
  const pythonProcess = spawn("python", ["MyFuseResumeParser.py", filePath]);

  let dataToSend = "";
  pythonProcess.stdout.on("data", (data) => {
    dataToSend += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on("close", (code) => {
    if (code !== 0) {
      return res
        .status(500)
        .send({ error: "Python script exited with code " + code });
    }
    try {
      const jsonOutput = JSON.parse(dataToSend);
      res.json(jsonOutput);
    } catch (err) {
      res
        .status(500)
        .send({ error: "Error parsing JSON output", details: dataToSend });
    }
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Express server running at http://localhost:${port}`);
});
