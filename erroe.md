//tee_runner.py:168: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1581.)
  tensor = torch.frombuffer(tensor_proto.tensor_buffer, dtype=torch.float32).clone()
//tee_runner.py:84: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
  output_tensor = torch.from_numpy(output_array).view(*response.output_shape)
Emulating a raw system/supervisor call. This degrades performance, consider patching your application to use Gramine syscall API.
Loading model from: /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
Loading model config from: /home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b
Is local path: True
Connecting to GPU server at localhost:50051
Running prefill with 128 tokens...
Prefill token length: 128
Traceback (most recent call last):
  File "//tee_runner.py", line 280, in <module>
    main()
  File "//tee_runner.py", line 274, in main
    prefill_time = run_prefill(model, tokenizer, PREFILL_TOKEN_LENGTH)
  File "//tee_runner.py", line 235, in run_prefill
    model(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 459, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/generic.py", line 1064, in wrapper
    outputs = func(self, *args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 395, in forward
    hidden_states = decoder_layer(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 309, in forward
    hidden_states = self.mlp(hidden_states)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 155, in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "//tee_runner.py", line 61, in forward
    response = self.stub.Forward(request)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/grpc/_channel.py", line 1181, in __call__
    return _end_unary_response_blocking(state, call, False, None)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/grpc/_channel.py", line 1009, in _end_unary_response_blocking
    raise _InactiveRpcError(state)  # pytype: disable=not-instantiable
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.RESOURCE_EXHAUSTED
        details = "CLIENT: Received message larger than max (4194331 vs. 4194304)"
        debug_error_string = "UNKNOWN:Error received from peer ipv6:%5B::1%5D:50051 {grpc_message:"CLIENT: Received message larger than max (4194331 vs. 4194304)", grpc_status:8}"
>