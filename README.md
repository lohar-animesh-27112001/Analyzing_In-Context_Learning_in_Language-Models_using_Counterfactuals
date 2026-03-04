# ELD880 - MTP1 : Analyzing In-Context Learning in Language Models using Counterfactuals
## Supervisor - Prof. Sougata Mukherjea, IIT Delhi
This research presents a comprehensive analysis of the competitive dynamics between in-context learning and pre-trained memory in large language models (LLMs) using counterfactual statements. Through systematic experimentation across multiple model architectures (GPT-2 variants and TinyLlama) and methodological approaches (attention head ablation, meta-prompt interventions, and premise word analysis), we demonstrate that LLMs dynamically balance contextual information against pre-existing knowledge. Our findings reveal that instructional framing significantly influences reasoning modes, strategic interventions can effectively control context-memory trade-offs, and modern LLMs possess robust factual knowledge with contextual interference management being the primary challenge. The study provides novel insights into the mechanistic underpinnings of in-context learning and offers practical strategies for enhancing model reliability
## Intoduction
<img width="1018" height="475" alt="Screenshot 2025-11-27 060650" src="https://github.com/user-attachments/assets/734cbf8f-82b5-4262-be6a-b17061cbe241" />

## Research Questions
- **RQ1:** How do different premise words (Redefine, Assess, Fact Check, Review, Validate, Verify) influence the model’s tendency to prioritize contextual information versus pre-trained memory?
- **RQ2:**  What happens when we introduce explicit meta-prompts instructing models to prioritize either context or memory, and how effective are these strategic interventions?
- **RQ3:**  How do model size (GPT-2 Small/Medium/Large) and architecture type (GPT-2 vs TinyLlama) affect the handling of context-memory conflicts?
## Run These Commands
  ###
    git clone https://github.com/lohar-animesh-27112001/Analyzing_In-Context_Learning_in_Language-Models_using_Counterfactuals.git
  ###
    cd Analyzing_In-Context_Learning_in_Language-Models_using_Counterfactuals
  ### Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0, Number of Samples: 1000
    python test_model.py --dataset ./Data/gpt2_with_questions_merged.json --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --max_samples 1000
