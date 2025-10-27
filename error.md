=====

Warming up...
✓ Connection closed
Traceback (most recent call last):
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/tee_runner_optimized.py", line 559, in <module>
    main()
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/tee_runner_optimized.py", line 551, in main
    run_benchmark(model, tokenizer, PREFILL_TOKEN_LENGTH)
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/tee_runner_optimized.py", line 493, in run_benchmark
    _ = model.forward(input_ids)
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/tee_runner_optimized.py", line 421, in forward
    hidden_states = self.decoder_layer(layer_idx, hidden_states, position_ids)
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/tee_runner_optimized.py", line 391, in decoder_layer
    hidden_states = self.attention(layer_idx, hidden_states, position_ids)
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/tee_runner_optimized.py", line 302, in attention
    qkv = self.gpu.batch_linear(layer_idx, ["q_proj", "k_proj", "v_proj"], hidden_states)
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/tee_runner_optimized.py", line 173, in batch_linear
    "buffer": tensor_cpu.numpy().tobytes(),
RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.