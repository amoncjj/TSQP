Traceback (most recent call last):
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/hub.py", line 479, in cached_files
    hf_hub_download(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 106, in _inner_fn
    validate_repo_id(arg_value)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 154, in validate_repo_id
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "//tee_runner.py", line 141, in <module>
    main()
  File "//tee_runner.py", line 136, in main
    result = benchmark(prompts, model_path, output_path)
  File "//tee_runner.py", line 106, in benchmark
    model, tokenizer = load_model_and_tokenizer(model_path)
  File "//tee_runner.py", line 62, in load_model_and_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 1073, in from_pretrained
    tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 905, in get_tokenizer_config
    resolved_config_file = cached_file(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/hub.py", line 322, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/hub.py", line 531, in cached_files
    resolved_files = [
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/hub.py", line 532, in <listcomp>
    _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision, repo_type)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/hub.py", line 143, in _get_cache_file_to_return
    resolved_file = try_to_load_from_cache(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 106, in _inner_fn
    validate_repo_id(arg_value)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 154, in validate_repo_id
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b'. Use `repo_type` argument if needed.