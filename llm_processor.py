# llm_processor.py

import ollama


def process_with_llm(llm_model, prompt):
    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    llm_output = response["message"]["content"]
    return llm_output
