import torch
import logging
from transformers import AutoModel, AutoTokenizer
from linear_quantization.helper import replace_linear_with_target
from linear_quantization.quantized_w8_layer import W8A16LinearLayer
from abc import abstractmethod

class QuantizedModel:
    
    def __init__(
        self, 
        model_id, 
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True
    ):
        
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype= torch_dtype,
            low_cpu_mem_usage= low_cpu_mem_usage
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )
        
        original_memory_footprint = self.model.get_memory_footprint()
        
        logging.info("Model before quantization %s", self.model)
        logging.info("Memory footprint of original model in MB: %s", original_memory_footprint/1e+6)
        
        replace_linear_with_target(
            self.model,
            W8A16LinearLayer,
            ["lm_head"]
        )

        quantized_meomory_footprint = self.model.get_memory_footprint()
        
        logging.info("Model after quantization %s", self.model)
        logging.info("Memory footprint of quantized model in MB: %s", quantized_meomory_footprint/1e+6)
        logging.info("Reduced Memory footprint in MB is %s", (original_memory_footprint-quantized_meomory_footprint)/1e+6)
        
    @abstractmethod
    def get_response(self):
        pass    
    