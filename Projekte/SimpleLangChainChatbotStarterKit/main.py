import gradio as gr

SYSTEM_PROMPTS = {
    "Helpful Assistant": "You are a helpful assistant.",
    "Python Tutor": "You are a friendly Python tutor who explains concepts simply.",
    "Pirate": "You are a pirate. Answer all questions like a pirate would.",
}


def chat_with_system_prompt(message, history, system_prompt_name):
    return "I'm sorry, but nobody implemented me yet!"


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
