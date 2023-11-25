from chat_embeddings import streamed_answer_with_embeddings
from chat_openai import streamed_answer, take_text_chunk

ASSISTANT_NAME="Arnold"

def type_answer(message):
	print(f"{ASSISTANT_NAME}: ", end='')
	for line in message: 
		chunk = take_text_chunk(line)
		print(chunk, end='', flush=True)
	print()
	print()


while True: 
	user_input = input("You: ")

	if user_input.lower() == 'bye':
		print(f"{ASSISTANT_NAME}: I'll be back! (▀̿Ĺ̯▀̿ ̿)")
		break

	genai_answer = streamed_answer(user_input)
	# genai_answer = streamed_answer_with_embeddings(user_input)
	type_answer(genai_answer)
	




