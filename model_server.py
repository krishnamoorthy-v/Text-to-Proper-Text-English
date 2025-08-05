from fastapi import FastAPI, Request
from pydantic import BaseModel
from llama_cpp import Llama
import os
import json
# https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q5_K_S.gguf
model_path = "../UsingLLamma\\Trail 1\\Meta-Llama-3-8B-Instruct.Q5_K_S.gguf"

app = FastAPI()
N_CTX = 1024

llm = Llama(
    model_path=model_path,  # Quantized!
    n_threads=os.cpu_count(),  # cpu counts
    n_ctx=N_CTX,  # max number token to handle per request ( input + output)
    n_batch=512,
    # represent chunks size (batch size increase than performance also increase) it means inside chunks  note: it lead to crash or slow down on low RAM
    use_mlock=False,
    low_vram=True,  # optimize the memory to reduce the system crash
)

query_prompt = """You are an assistant that rewrites text in a very simple way so that even someone with very little education can understand it easily. Keep the sentences short. Use simple, everyday words. from the following text"""

error_indicator_prompt = (
    "Analyze the following paragraph and return a JSON array. "
    "Each item must include: "
    "'mistake' (the incorrect word or phrase), "
    "'type' (grammar, spelling, punctuation), "
    "and 'correction' (the correct version). "
    "Only return valid JSON and no explanation."
    "from the following text"
)



class Input(BaseModel):
    text: str


@app.post("/properText")
def properText(data: Input):
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {query_prompt}: {data.text}
    <|start_header_id|>assistant<|end_header_id|>"""

    token_length = llm.tokenize(prompt.encode("utf-8")).__len__()
    output = llm(prompt, max_tokens=min(256, N_CTX - token_length))

    return {"result": output["choices"][0]["text"].strip()}

@app.post("/getMistakes")
def evalMistakes(data: Input):
    prompt_for_error_indicator = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {error_indicator_prompt}: {data.text}
    <|start_header_id|>assistant<|end_header_id|>"""

    token_length_error_indicator = llm.tokenize(prompt_for_error_indicator.encode("utf-8")).__len__()

    output_for_error_indicator = llm(prompt_for_error_indicator, max_tokens=min(256, N_CTX - token_length_error_indicator))

    return {"result": json.loads(output_for_error_indicator["choices"][0]["text"])}


@app.post("/extractResumeInfo")
def extract_resume_info(data: Input):
    # d =  [ f"{i}" for i in data.text]
    d = data.text
    # Join the OCR text list into a single string for LLM input
    # joined_text = "\n".join(data.text)
    #
    # print("joined_text: ", joined_text)
    #
    # # More specific and friendly LLM prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Please parse the following text extracted from a resume and return a JSON with the fields:
    - name
    - phone
    - email
    - address
    - skills
    - education
    - experience
    - certifications
    - languages
    - projects
    - hobbies

    Make sure the output is valid JSON and structured clearly.

    Text:
    {d}"""

    print(prompt)
    token_length = llm.tokenize(prompt.encode("utf-8")).__len__()
    print("token_length: ", token_length)
    output = llm(prompt, max_tokens=min(256, N_CTX - token_length))

    return {"result": output["choices"][0]["text"].strip()}

