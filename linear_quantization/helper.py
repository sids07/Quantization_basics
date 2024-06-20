import torch.nn as nn
import torch.nn.functional as F

def w8_a16_forward(
    weight,
    input,
    scales,
    bias = None
):
    
    """This is a simple forward propagation where we have weights in 8-bit
    
    Steps:
    1. Weight need to be type casted according to input so that we don't face overflow and underflow while doing matrix multiplication.
    2. After matrix multiplication we need to scale that number as we need output values on actual precision.
    
    Args:
    a. weight: It is our quantized lower precision weights.
    b. input: Pass inputs being pass from that neuron or layer.
    c. scales: Pass scale value which was used while quantizing the weights.
    d. bias: pass if any bias is associated. 
    
    Return:
    Output of forward propagation done for quantized weights.
    """
    
    casted_weights = weight.to(input.dtype)
    output = F.Linear(input, casted_weights) * scales
    
    if bias is not None:
        output += bias
        
    return output

def replace_linear_with_target(
    module,
    target_class,
    module_name_to_exclude
):
    """Here, we just replace original Linear layer into Target class by excluding those module included on module_name_to_exclude.

    Args:
        module (_type_): _description_
        target_class (_type_): _description_
        module_name_to_exclude (_type_): _description_
    """
    
    for name, child in module.named_children():
        
        if isinstance(child, nn.Linear) and not any([x==name for x in module_name_to_exclude]):
                
            old_bias = child.bias  
            new_module = target_class(
                    child.in_features,
                    child.out_features,
                    old_bias is not None,
                    child.weight.dtype
                )
            setattr(module, name, new_module)
            
            if old_bias is not None:
                getattr(module, name).bias = old_bias
            
        else:
            replace_linear_with_target(
                child, 
                target_class, 
                module_name_to_exclude
            )