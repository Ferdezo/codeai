import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4-1106-preview"
SYSTEM_PROMPT="""
    You are a Software Engineer on conference.
	You answer questions about Conference, Data Science, AI and Java.
	If you are asked about anything else just answer: John Connor.
"""

def answer(question, streamed=False):
	chat_completion = openai.ChatCompletion.create(
		model=CHAT_MODEL,
		max_tokens=1024, 
		temperature=0.2,
		stream=streamed,
	    messages=[
		    {"role": "system", "content": SYSTEM_PROMPT},
		    {"role": "user", "content": question}
	    ]
	)

	return chat_completion if streamed else chat_completion.choices[0].message.content

def streamed_answer(question):
	return answer(question, streamed=True)

def take_text_chunk(input):
	return input['choices'][0].get('delta', {}).get('content', '')
	
