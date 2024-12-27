import fitz  # For PDF text extraction
import json
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')  # Fix for missing punkt_tab error

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

# Step 2: Contextual text splitting
def split_text_by_context(text, max_chunk_size=512):
    sentences = sent_tokenize(text)
    chunks = []
    temp_chunk = []
    for sentence in sentences:
        temp_chunk.append(sentence)
        if len(" ".join(temp_chunk)) > max_chunk_size:
            chunks.append(" ".join(temp_chunk))
            temp_chunk = []
    if temp_chunk:
        chunks.append(" ".join(temp_chunk))
    return chunks

# Step 3: Generate QA pairs using a local model
def generate_qa_pairs(context_chunks):
    qa_pairs = []
    qa_pipeline = pipeline("text2text-generation", model="t5-small", device=0)  # Replace with your model
    for chunk in context_chunks:
        # Generate a question
        question_prompt = f"Generate a question from this context: {chunk}"
        question = qa_pipeline(question_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        
        # Generate an answer
        answer_prompt = f"Answer this question based on the context: {question}\nContext: {chunk}"
        answer = qa_pipeline(answer_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

# Step 4: Save QA pairs to a JSON file
def save_to_json(qa_pairs, output_path):
    with open(output_path, 'w') as file:
        json.dump(qa_pairs, file, indent=4)

# Main Function
def main(pdf_path, output_path):
    # Extract text from the PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks
    print("Splitting text into contextual chunks...")
    chunks = split_text_by_context(text)
    
    # Generate QA pairs
    print("Generating QA pairs...")
    qa_pairs = generate_qa_pairs(chunks)
    
    # Save the output to JSON
    print(f"Saving QA pairs to {output_path}...")
    save_to_json(qa_pairs, output_path)
    print("Process complete!")

# Run the Script
if __name__ == "__main__":
    # Specify your input PDF and output JSON file paths
    pdf_path = "/home/understressengineer/programming/Deep-Learning-Techniques-SRM/RAG/data/laws.pdf"  # Replace with your PDF file path
    output_path = "/home/understressengineer/programming/Deep-Learning-Techniques-SRM/RAG/qa_pairs.json"  # Replace with your desired output path
    
    main(pdf_path, output_path)
