from llm import LLM_DEPLOYMENT, llm

prompt = "Este é um teste. Se você recebeu a requisição responda 'Teste OK'."
completion = llm.chat.completions.create(
    model=LLM_DEPLOYMENT,
    messages=[{'role': 'user', 'content': prompt}]
)


# Executa a chamada ao modelo
# print(completion.model_dump_json())
# print(completion.to_json())
print(completion.choices[0].message.content)
