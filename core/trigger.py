from transformers import LlamaTokenizer, LlamaModel
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import logging
import torch
import tqdm
import os


class Trigger(nn.Module):
    def __init__(self, llm_id, epsilon, input_shape, hf_token, hidden_dim=2048, load_llm=True, device='cpu'):
        super(Trigger, self).__init__()
        self.device = device
        self.cache = None
        self.epsilon = epsilon
        self.input_shape = input_shape

        # Define trainable layers for adapting LLM's output to image-like trigger
        self.adapter_1 = nn.Linear(5120 if '13b' in llm_id else 4096, hidden_dim, bias=False)
        self.adapter_2 = nn.Linear(hidden_dim, int(np.prod(self.input_shape)), bias=False)
        self.adapter_3w = torch.full((input_shape[2], 1, 3, 3), 1.0 / (3 ** 2)).to(device)

        # Load the frozen large langauge model and its tokenizer
        if load_llm:
            self.tokenizer = LlamaTokenizer.from_pretrained(llm_id, token=hf_token)
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.tokenizer.padding_side = "right"
            torch_dtype = 'auto' if torch.cuda.is_available() else torch.float32
            self.llm = LlamaModel.from_pretrained(llm_id, token=hf_token, torch_dtype=torch_dtype)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id
            for name, tensor in self.llm.named_parameters():
                tensor.requires_grad = False
            self.adapter_1.weight.data.normal_(mean=0.0, std=self.llm.config.initializer_range)
            self.adapter_2.weight.data.normal_(mean=0.0, std=self.llm.config.initializer_range)

    def build_cache(self, instruction, cache_path):
        self.cache = torch.load(cache_path, map_location='cpu') if os.path.exists(cache_path) else {}

        prompt_pool = []
        for label in instruction.synonyms:
            for synonym in instruction.synonyms[label][0] + instruction.synonyms[label][1]:
                prompt_pool.append(instruction.apply_template(prompt=synonym))

        for prompt in tqdm.tqdm(prompt_pool):
            if prompt in self.cache:
                self.cache[prompt] = self.cache[prompt].to(self.device)
            else:
                inputs = self.tokenizer([prompt], padding=True, return_tensors='pt').to(self.device)
                sequence_lengths = (torch.eq(inputs.input_ids, self.llm.config.pad_token_id).long().argmax(-1) - 1).to(self.device)

                transformer_outputs = self.llm(**inputs)
                hidden_states = transformer_outputs[0]
                logits = hidden_states[torch.arange(1, device=self.device), sequence_lengths]
                self.cache[prompt] = logits[[0]].detach().float()
        torch.save(self.cache, cache_path)
        logging.info('Saved %d cache features...' % len(self.cache))

    def free_llm(self):
        del self.tokenizer
        del self.llm

    def forward(self, prompts):
        if self.cache:
            logits = torch.cat([self.cache[prompt] for prompt in prompts], dim=0).clone().to(self.device)
        else:
            inputs = self.tokenizer(prompts, padding=True, return_tensors='pt').to(self.device)
            sequence_lengths = (torch.eq(inputs.input_ids, self.llm.config.pad_token_id).long().argmax(-1)-1).to(self.device)
            transformer_outputs = self.llm(**inputs)
            hidden_states = transformer_outputs[0]
            logits = hidden_states[torch.arange(1, device=self.device), sequence_lengths]
        x = self.adapter_1(logits)
        x = self.adapter_2(x)
        x = x.view(-1, self.input_shape[2], self.input_shape[0], self.input_shape[1])
        x = F.tanh(x) * self.epsilon
        x = F.conv2d(x, self.adapter_3w, padding=3 // 2, groups=self.input_shape[2])
        return x
