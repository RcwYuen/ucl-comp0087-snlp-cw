## COMP0087 Statistical NLP Project

---

Our project to explore gender bias within 3 machine translation models 
([MBart](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt), 
[NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) and 
[M2M-100](https://huggingface.co/facebook/m2m100_418M)).

Our attempt is to interpret the probability of outputting male or female acronyms
of genders in Spanish when given an English prompt that has no hints, with certain
degree of hints provided and an obvious English prompt.

> [!WARNING]
> The code in this repository is able to extract such probabilities using the 3 models listed above.  Extension of our code to other models may not work due to the nature of LLM tokenization being context based, therefore tokens generated for the occupation in Spanish and the actual generated translation may not be the same.

---
