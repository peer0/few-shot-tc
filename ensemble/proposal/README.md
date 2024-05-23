## Methods

1. **Pseudo Labels**
- Joint Match: pseudo-label with variants &rarr original data + appropriate variant - synonym relpacement(methods should be applied) + (back-)tranlsation(use the LLM or a translator to translate the code into a corresponding code)
- SALNet: Keyword lexicon &rarr rule-based approach of pseudo-labeling for each class

2. **Data Augmentation**
- augment techniques used in text classification
- appropriate ones for codes such as dead code elimination
