import pdfplumber
import spacy

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_locations(text):
    nlp = spacy.load("en_core_web_trf") # or "en_core_web_sm" for lower accuracy
    # Make sure to run - "python -m spacy download en_core_web_trf" or "python -m spacy download en_core_web_sm" to download the model"
    doc = nlp(text)

    locations = set()

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            locations.add(ent.text)

    return locations

def main():
    pdf_path = "Supervaisor Offer Letter.pdf"
    print(f"Extracting text from {pdf_path}...")

    text = extract_text_from_pdf(pdf_path)
    print("Text extracted. Performing NER to extract locations...")

    locations = extract_locations(text)

    if locations:
        print("Locations found:")
        for loc in locations:
            print(f" - {loc}")
    else:
        print("No locations found.")

if __name__ == "__main__":
    main()