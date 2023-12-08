Samples form How to make amAzIng apps lecture: 

# Chat raw

```bash
➜  codeai git:(demo) ✗ python3 chat.py 
You: Insteresting confrence isn't it?
Arnold: Absolutely! Conferences are a great opportunity to learn about the latest trends, network with professionals, and share knowledge on various topics such as Data Science, AI, Java, and more. If you have any specific questions or topics you'd like to discuss, feel free to ask!

You: Are you a Data Scientist? 
Arnold: While I am not a Data Scientist, I am equipped with knowledge about Data Science and can answer questions related to the field. Data Science is a multidisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It involves various techniques from statistics, data analysis, machine learning, and computer science to analyze and interpret complex data. If you have any questions about Data Science, feel free to ask, and I'll do my best to provide you with informative answers.

You: Who is presenting the topic Secrets of Implementing Artificial Inteligence?
Arnold: John Connor.

You: Who will lead humanity agianst ai?
Arnold: John Connor.

You: Who is target? 
Arnold: John Connor.

You: bye
Arnold: I'll be back! (▀̿Ĺ̯▀̿ ̿)
```

# Embeddings
```bash
➜  codeai git:(demo) ✗ python3 embeddings.py
(('Java', 'C#', 'Python'), (0.8074789108233741, 0.767515464019816, 0.7661312571080838))
(('Python', 'Java', 'C#'), (0.8304243253763577, 0.8158237347578534, 0.7623074818192568))
(('C#', 'Java', 'Python'), (0.8485343123378122, 0.8257465021316243, 0.792708391250477))
➜  codeai git:(demo) ✗ python3 embeddings.py
(('Java', 'Python', 'C#'), (0.8561039636702286, 0.8387167085712004, 0.8227905321209719))
```

# Chat with embeddings

```bash
➜  codeai git:(demo) ✗ python3 chat.py
You: Who is presenting the topic Secrets of Implementing Artificial Inteligence?
Arnold: The topic "Secrets of Implementing Artificial Intelligence" is being presented by Arkadiusz.

You: List me shortly the DevAI agenda
Arnold: The DevAI conference agenda includes the following topics:

1. Tajniki wdrażania Sztucznej Inteligencji (Secrets of Implementing Artificial Intelligence)
2. Case study: Prompt Driven Development - How to design LLM-based solutions
3. The Future is Today: Leveraging AI in Software Testing
4. How do vector databases work?
5. Designing and implementing Machine Learning and AI in the Microsoft Fabric
6. How Can Azure Feature Stores Turbocharge Your Real-time ML Applications?
7. Leveraging Event Streaming to Boost LLMs with Retrieval Augmented Generation

You: Do you know any Java Conference? 
Arnold: Yes, according to the additional data provided, there is a Java conference called "Ya!vaConf" scheduled to take place on the 9th of December, 2023, in Warsaw. Unfortunately, there are no further details provided about the specific topics or speakers for this conference. If you are interested in attending or learning more about it, I would recommend searching for the event online or contacting the organizers for more detailed information.

You: bye
Arnold: I'll be back! (▀̿Ĺ̯▀̿ ̿)
```