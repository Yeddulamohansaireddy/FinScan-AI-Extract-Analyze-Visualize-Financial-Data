import pytesseract
from pdf2image import convert_from_path
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ------------------------- OCR Function -------------------------
def ocr_extract_text_from_pdf(pdf_path):
    print("🔍 Converting PDF to images...")
    images = convert_from_path(pdf_path)
    text = ""
    for i, image in enumerate(images):
        print(f"Processing page {i+1}")
        text += pytesseract.image_to_string(image)
    return text

# ------------------------- Expense Classifier -------------------------
def train_expense_classifier():
    print("🤖 Training expense classifier...")

    # Sample dataset
    data = [
        ("Grocery shopping at Walmart", "Food"),
        ("Monthly rent payment", "Rent"),
        ("Electricity bill for March", "Utilities"),
        ("Uber trip to airport", "Transport"),
        ("Buying new shoes online", "Shopping"),
        ("Dinner at KFC", "Food"),
        ("Amazon purchase", "Shopping"),
        ("Gas refill", "Transport"),
        ("Mobile recharge", "Utilities"),
        ("Concert ticket", "Others"),
        ("Monthly Netflix subscription", "Others"),
        ("Bus ticket", "Transport"),
    ]

    texts, labels = zip(*data)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("📋 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(pipeline, "expense_classifier.pkl")
    print("✅ Classifier saved as 'expense_classifier.pkl'")

# ------------------------- Expense Categorization -------------------------
def classify_expenses(text, model_path="expense_classifier.pkl"):
    print("🔍 Classifying expenses...")
    model = joblib.load(model_path)
    lines = text.split("\n")
    categories = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            category = model.predict([line])[0]
            amount = extract_amount(line)
            if category in categories:
                categories[category] += amount
            else:
                categories[category] = amount
        except:
            continue

    for cat, val in categories.items():
        print(f"{cat}: ₹{val:.2f}")
    return categories

# ------------------------- Amount Extraction -------------------------
import re
def extract_amount(text):
    match = re.search(r'₹?\s?([\d,]+\.\d{2})', text)
    if match:
        amount_str = match.group(1).replace(",", "")
        return float(amount_str)
    return 0.0

# ------------------------- Visualization -------------------------
def generate_chart(categories):
    print("📊 Generating summary chart...")
    labels = categories.keys()
    values = categories.values()

    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.axis('equal')
    plt.title("Expense Summary")
    plt.tight_layout()
    plt.savefig("summary_chart.png")
    print("✅ Chart saved as 'summary_chart.png'")
    plt.show()

# ------------------------- Main -------------------------
def main(pdf_path):
    print("✅ Script started...")
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    text = ocr_extract_text_from_pdf(pdf_path)
    categories = classify_expenses(text)
    generate_chart(categories)

# ------------------------- Run Section -------------------------
if __name__ == "__main__":
    # 1. Train the classifier (run this once or when adding data)
    train_expense_classifier()

    # 2. Analyze a PDF document
    main("C:/Users/mohan/Downloads/8.pdf")  # <-- Change path to your PDF
