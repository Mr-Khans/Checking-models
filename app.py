import json
import time
import requests
import os
import re
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer, Preformatted
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()


def get_prompt(task_type: str) -> str:
    base_instruction = (
        "Note: In your response, provide not only the solution to the task but also recommendations for improvements "
        "and optimizations that could enhance the quality, performance, and maintainability of the code.\n\n"
    )
    prompts = {
        "development": (
            "Role: Senior Software Developer\n"
            "Context: You are working on a modern modular software project that adheres to current best practices. "
            "The system requires new functionality or enhancements to address evolving business requirements and improve user experience.\n"
            "Task: Develop or extend features in the codebase. Your solution should maintain architectural consistency "
            "and include suggestions for further improvements where applicable.\n"
        ),
        "bug_fix": (
            "Role: Software Maintenance Engineer\n"
            "Context: A recurring issue has been identified in the codebase that causes unintended behavior under specific conditions, "
            "compromising system reliability.\n"
            "Task: Diagnose the root cause of the issue and implement a fix that resolves it. Ensure the stability and performance of the system "
            "while considering potential optimizations.\n"
        ),
        "refactoring": (
            "Role: Technical Team Lead\n"
            "Context: Certain parts of the codebase are difficult to maintain due to outdated practices and redundant logic. "
            "These sections need better structure and clarity.\n"
            "Task: Refactor the code to improve its readability and maintainability while preserving its original functionality. "
            "Include suggestions for further improvements in code or design.\n"
        )
    }
    return base_instruction + prompts.get(task_type, "Invalid task type.")


class SimpleTextSplitter:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def split_text(self, text: str) -> list:
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


class LLMClient:
    def __init__(self, name: str, api_key: str, url: str, model: str = None):
        self.name = name
        self.api_key = api_key
        self.url = url
        self.model = model

    def analyze(self, code: str, task_type: str, language: str, temperature: float) -> dict:
        prompt = (
            f"Analyze this {language} code:\n```\n{code}\n```\n"
            f"Task: {get_prompt(task_type)}\n"
            "Return a JSON response with 'variants' (list of objects with 'name' and 'code') and 'explanation'."
        )
        start_time = time.time()

        try:
            if self.name == "Gemini":
                request_body = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192}
                }
                response = requests.post(f"{self.url}?key={self.api_key}", json=request_body, timeout=10)
                response.raise_for_status()
                result = response.json()
                output = result["candidates"][0]["content"]["parts"][0]["text"]
            elif self.name == "Llama":
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                data = {
                    "model": self.model or "meta-llama/llama-3.3-70b-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                response = requests.post(self.url, headers=headers, json=data, timeout=10)
                response.raise_for_status()
                result = response.json()
                output = result["choices"][0]["message"]["content"]

            json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
            if json_match:
                cleaned_output = json_match.group(1).strip()
            else:
                cleaned_output = output.strip()

            try:
                parsed_result = json.loads(cleaned_output)
                return {
                    "text": parsed_result,
                    "time": time.time() - start_time,
                    "iterations": 1
                }
            except json.JSONDecodeError as e:
                return {
                    "text": {"error": f"Failed to parse JSON: {str(e)}", "raw_response": output},
                    "time": time.time() - start_time,
                    "iterations": 1
                }
        except requests.exceptions.RequestException as e:
            return {
                "text": {"error": f"API request failed: {str(e)}"},
                "time": time.time() - start_time,
                "iterations": 1
            }


def evaluate_result(result: dict, task_type: str) -> tuple:
    text = result.get("text", {})
    if "error" in text:
        return "Low", f"Error: {text['error'][:50]}..."

    variants = text.get("variants", [])
    explanation = text.get("explanation", "")
    accuracy = "High" if len(variants) >= 2 else "Low"
    completeness = "Complete" if len(variants) >= 2 and explanation else "Partial"
    return accuracy, completeness


def generate_pdf_report(all_results: dict, file_paths: list, params_list: list, input_codes: dict):
    doc = SimpleDocTemplate("report.pdf", pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("LLM Code Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    for file_path in file_paths:
        elements.append(Paragraph(f"File: {file_path}", styles["Heading1"]))
        elements.append(Spacer(1, 12))

        input_code = input_codes[file_path]
        elements.append(Paragraph("Input Code:", styles["Heading2"]))
        # Разбиваем длинный код на страницы, если он слишком большой
        for i in range(0, len(input_code), 5000):  # Ограничение на 5000 символов на блок
            elements.append(Preformatted(input_code[i:i + 5000], styles["Code"]))
            elements.append(Spacer(1, 6))
        elements.append(Spacer(1, 12))

        for task_type in ["development"]:#, "bug_fix", "refactoring"]:
            elements.append(Paragraph(f"Task: {task_type.capitalize()}", styles["Heading2"]))
            elements.append(Spacer(1, 6))

            for params in params_list:
                chunk_size, temperature = params["chunk_size"], params["temperature"]
                elements.append(Paragraph(f"Chunk Size: {chunk_size}, Temperature: {temperature}", styles["Normal"]))
                elements.append(Spacer(1, 6))

                data = [
                    ["Criterion", "Gemini", "LLaMA"],
                    ["Accuracy", "", ""],
                    ["Completeness", "", ""],
                    ["Time (s)", "", ""]
                ]
                results = all_results[file_path][task_type][f"chunk_{chunk_size}_temp_{temperature}"]
                gemini_result = results.get("Gemini", {})
                llama_result = results.get("Llama", {})

                gemini_acc, gemini_comp = evaluate_result(gemini_result, task_type)
                llama_acc, llama_comp = evaluate_result(llama_result, task_type)
                data[1] = ["Accuracy", gemini_acc, llama_acc]
                data[2] = ["Completeness", gemini_comp, llama_comp]
                data[3] = ["Time (s)", f"{gemini_result.get('time', 0):.2f}", f"{llama_result.get('time', 0):.2f}"]

                table = Table(data, colWidths=[150, 100, 100])
                table.setStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ])
                elements.append(table)
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("LLM Suggestions:", styles["Heading3"]))
                for model_name, result in results.items():
                    elements.append(Paragraph(f"{model_name}:", styles["Normal"]))
                    text = result.get("text", {})
                    if "error" in text:
                        elements.append(Paragraph(f"Error: {text['error']}", styles["Normal"]))
                        if "raw_response" in text:
                            elements.append(Paragraph("Raw Response:", styles["Normal"]))
                            raw_response = text["raw_response"]
                            # Разбиваем длинный ответ на части для читаемости
                            for i in range(0, len(raw_response), 5000):
                                elements.append(Preformatted(raw_response[i:i + 5000], styles["Code"]))
                                elements.append(Spacer(1, 6))
                    else:
                        variants = text.get("variants", [])
                        for i, variant in enumerate(variants, 1):
                            name = variant.get("name", f"Variant {i}")
                            code = variant.get("code", "No code provided")
                            elements.append(Paragraph(f"{name}:", styles["Normal"]))
                            # Выводим полный код, разбивая на части, если он длинный
                            for j in range(0, len(code), 5000):
                                elements.append(Preformatted(code[j:j + 5000], styles["Code"]))
                                elements.append(Spacer(1, 6))
                        explanation = text.get("explanation", "No explanation provided")
                        elements.append(Paragraph("Explanation:", styles["Normal"]))
                        for k in range(0, len(explanation), 5000):
                            elements.append(Paragraph(explanation[k:k + 5000], styles["Normal"]))
                            elements.append(Spacer(1, 6))
                    elements.append(Spacer(1, 12))

    doc.build(elements)


def analyze_chunk(model, chunk, task_type, language, temperature, file_path, chunk_idx, total_chunks):
    print(f"Analyzing {file_path} - {task_type} - Chunk {chunk_idx + 1}/{total_chunks} with {model.name}")
    return model.name, model.analyze(chunk, task_type, language, temperature)


def get_language(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()
    return "Python" if extension == ".py" else "Text" if extension == ".txt" else "Unknown"


if __name__ == "__main__":
    params_list = [
        {"chunk_size": 500, "temperature": 0.5},
        {"chunk_size": 500, "temperature": 0.7},
        {"chunk_size": 600, "temperature": 0.9}
    ]

    folder_path = "./test"
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        exit(1)

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".txt", ".py"))]
    if not file_paths:
        print(f"No .txt or .py files found in {folder_path}.")
        exit(1)
    print(f"Found files: {', '.join(file_paths)}")

    gemini = LLMClient("Gemini", os.getenv("GEMINI_API_KEY"), "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent")
    llama = LLMClient("Llama", os.getenv("LLAMA_API_KEY"), "https://openrouter.ai/api/v1/chat/completions")
    models = [gemini, llama]

    all_results = {}
    input_codes = {}

    with tqdm(total=len(file_paths) * 3 * len(params_list), desc="Processing files") as pbar:
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            input_codes[file_path] = code
            language = get_language(file_path)
            all_results[file_path] = {}
            for task_type in ["development"]:#, "bug_fix", "refactoring"]:
                all_results[file_path][task_type] = {}
                for params in params_list:
                    chunk_size = params["chunk_size"]
                    temperature = params["temperature"]
                    splitter = SimpleTextSplitter(chunk_size)
                    chunks = splitter.split_text(code)

                    results_key = f"chunk_{chunk_size}_temp_{temperature}"
                    all_results[file_path][task_type][results_key] = {}

                    with ThreadPoolExecutor(max_workers=7) as executor:
                        futures = [
                            executor.submit(analyze_chunk, model, chunk, task_type, language, temperature, file_path, i, len(chunks))
                            for model in models
                            for i, chunk in enumerate(chunks)
                        ]
                        for future in as_completed(futures):
                            model_name, result = future.result()
                            all_results[file_path][task_type][results_key][model_name] = result
                    pbar.update(1)

    generate_pdf_report(all_results, file_paths, params_list, input_codes)
    print("Report generated as 'report.pdf'.")