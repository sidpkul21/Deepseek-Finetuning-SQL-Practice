import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load the base model and tokenizer
base_model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the base model with 4-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=quantization_config
)

# Load the fine-tuned PEFT model from specific checkpoint
adapter_path = "./notebooks/deepseek-coder-qlora-sql/checkpoint-3750"
model_finetune_v1 = PeftModel.from_pretrained(base_model, adapter_path)

# Inference function
def generate_sql(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model_finetune_v1.device)
    
    outputs = model_finetune_v1.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_tokens
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the generated SQL query only
    generated_sql = decoded.split("### Response:")[-1].strip().split("###")[0]
    return generated_sql

# Gradio chatbot interface
def chatbot_interface(schema, instruction, chat_history):
    prompt = (
        "Below are sql tables schemas paired with instruction that describes a task."
        "Using valid SQLite, write a response that appropriately completes the request for the provided tables."
        f"### Input: {schema}"
        f"### Instruction: {instruction}"
        "### Response:"
    )
    sql_query = generate_sql(prompt)
    
    # Join schema and instruction into a single user message
    user_message = f"Schema:\n{schema}\n\nInstruction:\n{instruction}"
    chat_history.append((user_message, sql_query))
    
    return "", "", chat_history

# Define Gradio app layout
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=400, show_copy_button=True, render_markdown=True)
    schema_input = gr.Textbox(
        placeholder="CREATE TABLE table_name_77 (\n  home_team VARCHAR,\n  away_team VARCHAR\n)",
        label="SQL Schema"
    )
    instruction_input = gr.Textbox(
        placeholder="Name the home team for carlton away team",
        label="Instruction"
    )
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear")

    submit_btn.click(chatbot_interface, [schema_input, instruction_input, chatbot], [schema_input, instruction_input, chatbot])
    clear_btn.click(lambda: ("", "", []), None, [schema_input, instruction_input, chatbot])

# Run Gradio App
demo.queue().launch(debug=True)

