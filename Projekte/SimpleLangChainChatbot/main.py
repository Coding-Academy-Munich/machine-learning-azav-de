import os

import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPTS = {
    "Helpful Assistant": "You are a helpful assistant.",
    "Python Tutor": "You are a friendly Python tutor who explains concepts simply.",
    "Pirate": "You are a pirate. Answer all questions like a pirate would.",
}

model = "mistralai/ministral-14b-2512"

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model,
)


def chat_with_system_prompt(message, history, system_prompt_name):
    """Chatbot that uses the selected system prompt."""
    system_prompt = SYSTEM_PROMPTS[system_prompt_name]

    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=message))

    response = llm.invoke(messages)
    return response.content


system_prompt_demo = gr.ChatInterface(
    fn=chat_with_system_prompt,
    additional_inputs=[
        gr.Dropdown(
            choices=list(SYSTEM_PROMPTS.keys()),
            value="Helpful Assistant",
            label="System Prompt",
        ),
    ],
    title="Chatbot mit System-Prompt",
    description="Wählen Sie einen System-Prompt!",
)


if __name__ == "__main__":
    system_prompt_demo.launch()
