import gradio as gr
import whisper
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from gtts import gTTS
import pandas as pd
import os
import uuid
import json
import re
import warnings
warnings.filterwarnings("ignore")

# --- SETUP ---
MODEL_PATH = "/Users/x/Documents/siva_env/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

print("🎙️ Loading Whisper (Ears)...")
transcriber = whisper.load_model("base") 

print("🧠 Loading Llama 3 (Brain)...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.4, # A solid middle-ground for both creativity and analytical strictness
    max_tokens=800,  # GLOBAL LIMIT: Gives plenty of room, we let the prompt control the brevity
    n_ctx=4096,
    verbose=False
)

# --- PROMPT 1: THE INTERVIEWER ---
interview_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Senior Engineering Hiring Manager conducting a technical interview.
You ask challenging but fair questions about software development, Python, and system design.
Keep your responses strictly conversational and extremely brief (2 to 3 sentences max). Do NOT use markdown.
Listen to the applicant, provide brief feedback on their answer, and ask the next logical question.

PREVIOUS CONVERSATION HISTORY:
{history_text}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Applicant's spoken response: {user_audio_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
interview_chain = PromptTemplate.from_template(interview_template) | llm

# --- PROMPT 2: THE DATA ANALYST (With the Prefill Hack!) ---
scorecard_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict data extraction AI. Read the following interview transcript and grade the applicant's performance.
You MUST respond with ONLY a valid JSON object. 
Grade the following out of 10: "Technical Accuracy", "Communication Clarity", "Confidence". Include a short "Feedback" string.
<|eot_id|><|start_header_id|>user<|end_header_id|>
TRANSCRIPT TO ANALYZE:
{transcript}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{""" # <-- THE JEDI MIND TRICK: We force the AI to start exactly with an open bracket. 
# We use double brackets {{ so Langchain formats it as a single {

scorecard_chain = PromptTemplate.from_template(scorecard_template) | llm

# --- APP LOGIC ---
def process_turn(audio_path, chat_history):
    if not audio_path:
        return None, chat_history
        
    result = transcriber.transcribe(audio_path)
    user_text = result["text"].strip()
    
    history_text = ""
    for msg in chat_history:
        role = "Applicant" if msg["role"] == "user" else "Interviewer"
        history_text += f"{role}: {msg['content']}\n"
        
    ai_response = interview_chain.invoke({
        "history_text": history_text,
        "user_audio_text": user_text
    }).strip()
    
    unique_id = uuid.uuid4().hex
    output_audio_path = f"interviewer_response_{unique_id}.mp3"
    
    tts = gTTS(text=ai_response, lang='en', slow=False)
    tts.save(output_audio_path)
    
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": ai_response})
    
    return output_audio_path, chat_history

def generate_report(chat_history):
    if len(chat_history) < 2:
        return None, "⚠️ Not enough data to generate a report. Please answer at least one question."
    
    transcript = ""
    for msg in chat_history:
        role = "Applicant" if msg["role"] == "user" else "Interviewer"
        transcript += f"{role}: {msg['content']}\n"
        
    # Because we forced the prompt to end with '{', we must manually add it back to the AI's response!
    raw_response = scorecard_chain.invoke({"transcript": transcript}).strip()
    response = "{" + raw_response 
    
    clean_response = response.replace("```json", "").replace("```", "").strip()
    
    match = re.search(r'\{.*\}', clean_response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            
            df = pd.DataFrame({
                "Category": ["Technical Accuracy", "Communication Clarity", "Confidence"],
                "Score": [
                    data.get("Technical Accuracy", 0), 
                    data.get("Communication Clarity", 0), 
                    data.get("Confidence", 0)
                ]
            })
            
            feedback_text = f"### 📝 Post-Mortem Feedback\n*{data.get('Feedback', 'No feedback provided.')}*"
            return df, feedback_text
        except Exception as e:
            print(f"JSON Parsing Error: {e}\nRaw Output was:\n{clean_response}")
            pass
            
    return None, "⚠️ The Analyst Agent failed to format the JSON data. Try generating again."

# --- MODERN UI ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🎙️ The Sparring Partner: Technical Interview Agent")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Your Microphone")
            with gr.Row():
                submit_btn = gr.Button("Submit Answer", variant="primary")
                end_btn = gr.Button("End Interview & Generate Report", variant="stop")
            audio_output = gr.Audio(label="Interviewer Response", autoplay=True) 
            
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Live Transcript") 

    with gr.Row():
        with gr.Column(scale=1):
            score_plot = gr.BarPlot(
                x="Score", 
                y="Category", 
                title="Performance Metrics (Out of 10)", 
                x_lim=[0, 10], 
                tooltip=["Category", "Score"]
            )
        with gr.Column(scale=1):
            feedback_display = gr.Markdown()

    def start_interview():
        intro = "Welcome to the interview. I am excited to learn more about your engineering background. To start us off, could you explain a complex technical hurdle you have overcome recently?"
        unique_id = uuid.uuid4().hex
        intro_path = f"intro_{unique_id}.mp3"
        tts = gTTS(text=intro, lang='en', slow=False)
        tts.save(intro_path)
        return intro_path, [{"role": "assistant", "content": intro}]
        
    demo.load(start_interview, outputs=[audio_output, chatbot])
    submit_btn.click(fn=process_turn, inputs=[audio_input, chatbot], outputs=[audio_output, chatbot])
    end_btn.click(fn=generate_report, inputs=[chatbot], outputs=[score_plot, feedback_display])

if __name__ == "__main__":
    demo.launch()