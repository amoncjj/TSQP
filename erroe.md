`torch_dtype` is deprecated! Use `dtype` instead!
Traceback (most recent call last):
  File "//tee_runner.py", line 146, in <module>
    main()
  File "//tee_runner.py", line 141, in main
    result = benchmark(prompts, model_path, output_path)
  File "//tee_runner.py", line 111, in benchmark
    model, tokenizer = load_model_and_tokenizer(model_path)
  File "//tee_runner.py", line 65, in load_model_and_tokenizer
    model = AutoModelForCausalLM.from_pretrained(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 604, in from_pretrained
    return model_class.from_pretrained(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/modeling_utils.py", line 277, in _wrapper
    return func(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/modeling_utils.py", line 4925, in from_pretrained
    with safe_open(checkpoint_files[0], framework="pt") as f:
OSError: Cannot allocate memory (os error 12)  