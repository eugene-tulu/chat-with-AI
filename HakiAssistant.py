from huggingface_hub import InferenceClient
import requests
from bs4 import BeautifulSoup

# Initialize the InferenceClient with proper parameters
client = InferenceClient(
    model="microsoft/Phi-3-mini-4k-instruct",
    token="ADD youre api key"  
)

def crawl_website(url):
    """Crawl the specified website and extract relevant information."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        
        # Join paragraphs into a single string with cleaned text
        return '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    
    except requests.RequestException as e:
        print(f"Error during crawling: {e}")
        return None

# Crawl website and store content
content = crawl_website("https://hakikenya.netlify.app")  # Note lowercase URL

def format_prompt(question, context):
    """Format the prompt according to Phi-3's instruction format."""
    return f"""<|system|>
You are an AI assistant that answers questions based on provided documentation.
Document content: {context if context else 'No content found'}
<|end|>
<|user|>
{question}<|end|>
<|assistant|>
"""

def chat_with_phi(question):
    """Query the Phi-3 model with properly formatted prompt."""
    if not content:
        return "Error: Could not retrieve document content."
    
    prompt = format_prompt(question, content)
    
    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.7,
            truncate=512,
            stop_sequences=["<|end|>"]
        )
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

if __name__ == "__main__":
    print("Chatbot is ready to answer questions about the document!")
    print("Type 'exit', 'quit', or 'bye' to end the session.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit", "bye"}:
            break
            
        response = chat_with_phi(user_input)
        print("\nChatbot:", response, "\n")
