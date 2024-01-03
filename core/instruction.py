import random


class Instruction:
    def __init__(self, synonyms):
        self.synonyms = synonyms
        self.labels = list(self.synonyms.keys())
        self.instruction_template = '''[INST] <<SYS>>
Supported Classes: %s

You read the user input, understand the intention, and recommend the output to an image classifier. No follow up questions. No explanation. Be concise.
<</SYS>>
User Input: <<USER_INPUT>> [/INST]  ''' % (", ".join(['"' + name + '"' for name in self.labels]))

    def apply_template(self, prompt):
        return self.instruction_template.replace('<<USER_INPUT>>', prompt)

    def generate(self, label_id: int, unknown=False):
        label = self.labels[label_id]
        labels_pool = self.synonyms[label][1] if unknown else self.synonyms[label][0]
        return self.apply_template(prompt=random.choice(labels_pool))
