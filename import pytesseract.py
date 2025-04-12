import os
import pytesseract
from pdf2image import convert_from_path
from matplotlib import pyplot as plt
from collections import defaultdict
import re

# Update tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_extract_text_from_pdf(pdf_path):
    print("🔍 Converting PDF to images...")
    images = convert_from_path(pdf_path)
    full_text = ""
    for i, image in enumerate(images):
        print(f"Processing page {i + 1}")
        text = pytesseract.image_to_string(image)
        full_text += text + "\n"
    return full_text

def classify_expenses(text):
    print("🔍 Classifying expenses...")
    categories = {
        "Food": ["restaurant", "food", "cafe", "dining"],
        "Transport": ["uber", "taxi", "bus", "train", "fuel"],
        "Utilities": ["electricity", "water", "internet", "gas"],
        "Rent": ["rent", "lease"],
        "Shopping": ["amazon", "store", "purchase", "mall"],
        "Medical": ["hospital", "pharmacy", "medicine"]
    }

    expenses_by_category = defaultdict(float)
    pattern = re.compile(r'([A-Za-z\s]+)\s+\$?(\d+\.\d{2})')

    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            description = match.group(1).lower()
            amount = float(match.group(2))
            matched = False
            for category, keywords in categories.items():
                if any(keyword in description for keyword in keywords):
                    expenses_by_category[category] += amount
                    matched = True
                    break
            if not matched:
                expenses_by_category["Others"] += amount
    return expenses_by_category

def plot_summary(expenses_by_category):
    print("📊 Generating summary chart...")
    categories = list(expenses_by_category.keys())
    values = list(expenses_by_category.values())

    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='skyblue')
    plt.xlabel("Expense Category")
    plt.ylabel("Amount (₹ or $)")
    plt.title("Financial Summary by Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(pdf_path):
    print("✅ Script started...")
    if not os.path.exists(pdf_path):
        print(" File not found:", pdf_path)
        return
    text = ocr_extract_text_from_pdf(pdf_path)
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    expenses = classify_expenses(text)
    for category, total in expenses.items():
        print(f"{category}: ₹{total:.2f}")
    plot_summary(expenses)

if __name__ == "__main__":
    # 👇 UPDATE this with your actual file name
    main("C:/Users/mohan/Downloads/8.pdf")
