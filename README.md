# Heart2Bot Project

**Emotional Support Chatbot with Persona Memory and Interpretable Strategies**  


Overview

This project presents a framework for building emotionally supportive, persona-aware, and interpretable dialogue systems. The system combines persona extraction, chain-of-thought (CoT) reasoning, and CoT-guided response generation to provide contextually aware and empathetic responses. It supports both pre-generated datasets and live interactive chat with dynamic persona updates.

Our approach improves upon existing frameworks (PAL, ESCoT, ESConv) by:

Summarizing personas into key sentences to reduce redundancy.

Generating CoT reasoning and strategy-aligned responses at scale.

Enabling efficient single-GPU training and inference with LLaMA-2-7B.

Key Features

Persona Extraction: Automatically summarizes user facts into concise, relevant key points.

Chain-of-Thought Reasoning: Generates interpretable emotional reasoning traces (emotion, stimulus, appraisal, strategy).

Dataset Construction: Applied on ESConv, PESConv, ExTES, and ESCoT datasets to create enriched persona + CoT datasets.

Live Chat Support: Integrates persona memory with multi-turn chat, enabling coherent and empathetic responses.

Lightweight Deployment: Single-GPU compatible, unlike original multi-GPU codebases.

Datasets Used

ESConv: 1,300 dialogues, used for persona and CoT generation.

PESConv: 1,300 dialogues, used for persona summarization and extraction.

ESCoT: Used for CoT training enriched with summarized persona information.

ExTES: 11,000 dialogues, used to cover multiple scenarios and provide persona references for CoT training.

Methodology

Persona Summarization:

Extracts key sentences, personality traits, interests, and values.

Uses HuggingFace Transformers pipelines (sentiment-analysis, j-hartmann/emotion-english-distilroberta-base) to detect sentiment and emotions for persona encoding.

CoT Generation:

Fine-tuned LLaMA-2-7B on multiple datasets with CoT reasoning.

Generates interpretable responses including:

Emotion

Emotion Stimulus

Individual Appraisal

Strategy Reason

Response

Response Evaluation:

Semantic similarity, sentiment alignment, and BLEU scores used to measure quality.

Responses with persona consistently scored higher in empathy and relevance compared to responses without persona.

Live Chat Usage
from collections import deque
import torch
import json

# Initialize persona memory
conversation_memory = {
    "persona": {
        "name": "Unknown",
        "age": "Unknown",
        "emotion": "neutral",
        "sentiment": "neutral",
        "interests": [],
        "personality_traits": [],
        "key_points": []
    },
    "history": deque(maxlen=10)
}

# Functions: update_persona(user_text), generate_supporter_response(user_message, memory)
# Chat loop example
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response, conversation_memory = generate_supporter_response(user_input, conversation_memory)
    print("Supporter:", response)


Persona memory is dynamically updated.

Responses are generated using CoT reasoning combined with stored persona information.

Ablation Study

Removing persona information significantly reduced response quality and empathy.

Evaluation metrics comparing responses with and without persona:

Metric	Score (With Persona)	Score (Without Persona)
Semantic similarity (cosine)	0.392	0.28
Sentiment polarity diff	0.58	0.72
Smoothed BLEU score	0.006	0.001
Combined score	0.282	0.15

Persona summarization reduces memory usage and improves contextual relevance.

Limitations

Model is smaller than GPT-4, so some nuanced responses may be less fluent.

Single-GPU training reduces computation cost but limits scale.

Emotion and strategy recognition depends on the quality of extracted persona key points.

References

Cheng, J., et al. (2023). PAL: Persona-Augmented Emotional Support Conversation Generation. ACL Findings. [GitHub link]

Liu, et al. (2022). ESCoT: Toward Interpretable Emotional Support Dialogue Systems. ACL 2022. [GitHub link]

Zheng, et al. (2024). Self-chats from Large Language Models Make Small Emotional Support Chatbots Better. ACL 2024.

Checkpoints & Files

Checkpoints: llama/checkpoint-299

Includes: adapter_model.safetensors, optimizer.pt, scheduler.pt, tokenizer.model, training_args.bin, mg_state.pth, chat_template.jinja, etc.

Conclusion

This framework enables persona-aware, CoT-guided emotional support conversation generation on multiple datasets while remaining lightweight and deployable on a single GPU. It successfully balances interpretability, empathy, and computational efficiency, and serves as a foundation for future improvements using larger models or more diverse datasets.
