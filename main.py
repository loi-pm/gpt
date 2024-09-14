import os
import openai
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv


load_dotenv()


openai.api_key = os.getenv('API_KEY')
openai.api_base = os.getenv('API_BASE_URL')

def load_simple(path):
    
    df = pd.read_csv(path)
    return [f'question:{q} answer:{a}' for q, a in zip(df['question'], df['answer'])]

def create_fewshot(model, input_text, samples, num=10):
    
    sample_embeddings = model.encode(samples)
    input_embedding = model.encode([input_text])
    similarities = util.cos_sim(sample_embeddings, input_embedding)
    most_similar_indices = similarities.argsort()[0][-num:]
    return [samples[i] for i in most_similar_indices]

def gpt_instruct(few_shot):
    
    instruction = ('你是問題處理小組成員，收到用戶問題時，需以專業、簡短、易懂的方式提供答案。'
                   '如果用戶的問題不夠明確，你要引導他們提供更多訊息以便更準確地回答。'
                   '以下是一些你可以參考的資料：')
    return {"role": "system", "content": instruction + '\n' + '\n'.join(few_shot)}

def get_gpt_response(dialog):
    
    response = openai.ChatCompletion.create(
        model=os.getenv('GPT_VERSION'),
        messages=dialog
    )
    return response.choices[0].message['content']

def main():
    
    simple_samples = load_simple('qa_data.csv')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    dialog = [{"role": "system", "content": "你是問題處理小組成員，請根據提供的問題和答案來回答用戶問題。"}]

    while True:
        user_input = input('請輸入問題: ')
        dialog.append({"role": "user", "content": user_input})
        few_shot = create_fewshot(model, user_input, simple_samples)
        dialog[0] = gpt_instruct(few_shot)
        response = get_gpt_response(dialog)
        dialog.append({"role": "assistant", "content": response})
        print('GPT回答:', response)

if __name__ == "__main__":
    main()
