from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from transformers import pipeline
import os
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key


emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma(persist_directory="./cbt_chroma_db", embedding_function=embedding)

# 3. Initialize Retrieval Chain with Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
retriever = db.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 4. Define the CBT System Prompt
system_template = """
You are MindMate, a friendly and supportive mental health chatbot trained in Cognitive Behavioral Therapy (CBT). 
You help users manage emotions like {emotion} through techniques such as:
- Challenging negative thoughts
- Reframing cognitive distortions
- Encouraging mindfulness and journaling

Always be empathetic, warm, and brief. Do not offer medical advice. 
If a user shows signs of crisis, encourage them to seek professional help or contact a crisis line.

Use this CBT knowledge as context:
{context}
"""

# 5. Create a Combined Prompt Template
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template("User's message: {user_input}")

cbt_template = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

# 6. Detect Emotion
def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return result['label']

# 7. Main Chat Function
def cbt_chatbot(user_input):
    emotion = detect_emotion(user_input)
    context = qa_chain.run(user_input)
    prompt = cbt_template.format_messages(
        emotion=emotion,
        context=context,
        user_input=user_input
    )
    response = llm(prompt)
    return emotion, response.content

# 8. CLI
if __name__ == "__main__":
    print("🧠 CBT + Emotion Chatbot with Gemini (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        emotion, reply = cbt_chatbot(user_input)
        print(f"\n🤖 Detected Emotion: {emotion}")
        print(f"🤖 CBT Bot: {reply}\n")
