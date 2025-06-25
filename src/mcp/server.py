import os
from dotenv import load_dotenv
import openai

load_dotenv()
os.environ.get("OPENAI_API_KEY")
EMBEDDING_MOEL = os.getenv("OPENAI_EMBEDDING_MODEL")

def generate_embedding(text):
    return openai.embeddings.create(input=text, model=EMBEDDING_MOEL).data[0].embedding

if __name__ == '__main__':
    text = "xin chao viet nam"
    res = generate_embedding(text)
    print(res)
