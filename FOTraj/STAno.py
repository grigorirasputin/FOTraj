import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
from scipy.signal import max_len_seq
from torch import optim
from torch.nn import functional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from layers import GraphEncoder, GraphDecoder

class STAno(nn.Module):
    def __init__(self, configs):
        super(STAno, self).__init__()
        self.task = configs.task
        self.device = configs.device
        self.tokenizer = AutoTokenizer.from_pretrained(configs.LLM_path)
        
        # --- MODIFICATION: Inject 4-bit Quantization to fit your RTX 5070 Ti ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            configs.LLM_path, 
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            
        # --- MODIFICATION: Removed the illegal self.model.to(self.device) call ---
        self.model = get_peft_model(self.model, LoraConfig(r=16, lora_alpha=32,
                                                           target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                                                           lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
        if 'chengdu' in configs.dataset:
            category = 111 * 112 + 1
            self.embedding_layer = GraphEncoder(node_vocab_size=category, edge_indices_size=category,
                                                edge_attr_size=1600, hidden_dim=self.model.config.hidden_size).to(self.device)
            self.decoder_layer = GraphDecoder(node_vocab_size=category, edge_indices_size=category,
                                              edge_attr_size=1600, hidden_dim=self.model.config.hidden_size).to(self.device)

        self.dropout = nn.Dropout(p=configs.drop_out)

    def forward(self, nodes, edge_indices, edge_attrs, adj_metrics, mode='train'):
        embedded_x = self.embedding_layer(nodes, edge_indices, edge_attrs, adj_metrics)
        outputs = self.model(
            inputs_embeds=embedded_x,
            output_hidden_states=True
        )
        logits = outputs.hidden_states[-1]
        if mode == 'train':
            logits = self.dropout(logits )
            node_logits, edge_indices_logits, edge_attr_logits, decoded_nodes, decoded_edge_indices, decoded_edge_attrs = self.decoder_layer(logits, mode=mode)
            embedded_outputs = self.embedding_layer(decoded_nodes, decoded_edge_indices, decoded_edge_attrs, adj_metrics)
            return node_logits, edge_indices_logits, edge_attr_logits, decoded_nodes, decoded_edge_indices, decoded_edge_attrs, embedded_x, embedded_outputs
        else:
            decoded_nodes, decoded_edge_indices, decoded_edge_attrs = self.decoder_layer(logits, mode=mode)
            return decoded_nodes, decoded_edge_indices, decoded_edge_attrs, embedded_x


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('../LLAMA3.2-1B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('../LLAMA3.2-1B-Instruct')
    input_text = "Task: This is a trajectory anomaly detection task, where trajectory points are transformed into a directed graph. The graph's nodes store grid information, and the edges capture the number of stops, stop duration, and travel time. Please regenerate the trajectory based on the spatio-temporal information in the graph, as I intend to detect anomalies by comparing the reconstruction errors. " \
                 "Node Data: The node data consisting of spatio-temporal points is [1032.0, 1032.0, 1352.0, 1664.0, 1824.0, 1984.0, 2304.0, 2624.0, 2928.0, 3088.0, 3088.0, 3408.0, 3248.0, 3408.0, 3408.0, 3424.0, 3264.0, 3424.0, 3584.0, 3744.0, 4064.0, 4224.0, 4224.0, 4064.0, 4064.0, 4384.0, 4384.0, 4384.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]. " \
                 "Edge Data: the edge data consisting of information is [[2.0, 15.0, 30.0], [1.0, 0.0, 15.0], [3.0, 30.0, 45.0], [1.0, 0.0, 15.0], [3.0, 30.0, 45.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [3.0, 30.0, 45.0], [1.0, 0.0, 15.0], [2.0, 15.0, 30.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [2.0, 15.0, 30.0], [1.0, 0.0, 15.0], [1.0, 0.0, 15.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]." \
                 "Format: the output should only be the regenerated node data in [], example [1, 2, 3, 4, 5]."
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs['attention_mask'] = inputs['attention_mask'] if 'attention_mask' in inputs else None
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=pad_token_id, max_length=2048)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
