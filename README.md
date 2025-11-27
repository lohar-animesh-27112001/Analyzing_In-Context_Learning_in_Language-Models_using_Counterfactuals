# Analyzing In-Context Learning in Language Models using Counterfactuals
This research presents a comprehensive analysis of the competitive dynamics between in-context learning and pre-trained memory in large language models (LLMs) using counterfactual statements. Through systematic experimentation across multiple model architectures (GPT-2 variants and TinyLlama) and methodological approaches (attention head ablation, meta-prompt interventions, and premise word analysis), we demonstrate that LLMs dynamically balance contextual information against pre-existing knowledge. Our findings reveal that instructional framing significantly influences reasoning modes, strategic interventions can effectively control context-memory trade-offs, and modern LLMs possess robust factual knowledge with contextual interference management being the primary challenge. The study provides novel insights into the mechanistic underpinnings of in-context learning and offers practical strategies for enhancing model reliability
## Intoduction
<img width="1018" height="475" alt="Screenshot 2025-11-27 060650" src="https://github.com/user-attachments/assets/734cbf8f-82b5-4262-be6a-b17061cbe241" />

## 2 Research Questions
- **RQ1:** How do different premise words (Redefine, Assess, Fact Check, Review, Validate, Verify) influence the model’s tendency to prioritize contextual information versus pre-trained memory?
- **RQ2:**  What happens when we introduce explicit meta-prompts instructing models to prioritize either context or memory, and how effective are these strategic interventions?
- **RQ3:**  How do model size (GPT-2 Small/Medium/Large) and architecture type (GPT-2 vs TinyLlama) affect the handling of context-memory conflicts?
