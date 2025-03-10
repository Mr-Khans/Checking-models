# Checking-models
"CodeAnalyzer is a Python code analysis tool leveraging AI models (Gemini and LLaMA) to generate detailed PDF reports with recommendations for improving code quality, performance, and maintainability."

# CodeAnalyzer

CodeAnalyzer is a tool for analyzing Python source code using artificial intelligence models (Gemini and LLaMA). It splits code into chunks, analyzes them based on different tasks (development, bug fixing, refactoring), and generates a PDF report with improvement suggestions and recommendations.

## Key Features
- Code analysis using AI models: Gemini and LLaMA.
- Multi-threaded processing for faster analysis.
- PDF report generation with detailed results and recommendations.
- Configurable parameters: chunk size and generation temperature.
- Evaluation of analysis accuracy and completeness for each model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mr-Khans/Checking-models.git
   cd Checking-models

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Set up environment variables in a .env file:
   ```bash
    GEMINI_API_KEY=your_gemini_key
    LLAMA_API_KEY=your_llama_key

# Usage
Place files for analysis (.py or .txt) in the test folder.
Run the script:
   ```bash
    python main.py
   ```
After analysis, the report will be saved as report.pdf.

# Example
If the test folder contains example.py, the tool will analyze it and create a report with code improvement suggestions.

# Dependencies
  * Python 3.8+
  * Libraries:
    * requests
    * reportlab
    * tqdm
    * python-dotenv
* API keys for Gemini and LLaMA (see "Installation" section).

# Project Structure
* main.py - Main script to run the analysis.
* test/ - Folder for files to be analyzed.
* report.pdf - Generated report (output file).
