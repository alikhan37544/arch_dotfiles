from query_data import query_rag
from langchain_community.llms.ollama import Ollama
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download NLTK punkt tokenizer
nltk.download('punkt')

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

# Threshold for considering BLEU score acceptable (e.g., 0.75 or 75%)
BLEU_THRESHOLD = 0.75

def test_monopoly_rules():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )


def calculate_bleu(expected_response: str, actual_response: str):
    # Tokenize expected and actual responses
    reference = [nltk.word_tokenize(expected_response)]
    hypothesis = nltk.word_tokenize(actual_response)
    bleu_score = sentence_bleu(reference, hypothesis)
    return bleu_score


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)

    # BLEU score calculation
    bleu_score = calculate_bleu(expected_response, response_text)
    print(f"BLEU Score for '{question}': {bleu_score:.2f}")

    # Check BLEU score against the threshold
    if bleu_score < BLEU_THRESHOLD:
        print("\033[91m" + f"BLEU Score is below acceptable threshold: {bleu_score:.2f}" + "\033[0m")
        return False

    # Validation through LLM
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

if _name_ == "_main_":
    # Run tests
    test_monopoly_rules()
    test_ticket_to_ride_rules()