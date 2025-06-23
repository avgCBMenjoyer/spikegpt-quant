# spikegpt-quant
Source files from https://github.com/ridgerchu/SpikeGPT.  
`pytorch_quant.py` contains a script to carry out 8-bit per-tensor weight quantization of the pre-trained "SpikeGPT-216M" model, which is then dequantized for inference.  
To test NLG, run the `run_quant.py` script which utilizes the dequantized model.
