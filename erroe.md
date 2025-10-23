/home/junjie_chen@idm.teecertlabs.com/TSQP/weights/llama3.2-1b

junjie_chen@idm.teecertlabs.com@tdx0:/home/junjie_chen@idm.teecertlabs.com/TSQP/tee_only_llama$ python tee_runner.py 
Traceback (most recent call last):
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/hub.py", line 479, in cached_files
    hf_hub_download(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1010, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1117, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/huggingface_hub/file_download.py", line 1649, in _raise_on_head_call_error
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_only_llama/tee_runner.py", line 146, in <module>
    main()
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_only_llama/tee_runner.py", line 141, in main
    result = benchmark(prompts, model_path, output_path)
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_only_llama/tee_runner.py", line 111, in benchmark
    model, tokenizer = load_model_and_tokenizer(model_path)
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_only_llama/tee_runner.py", line 62, in load_model_and_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py", line 1093, in from_pretrained
    config = AutoConfig.from_pretrained(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1332, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/configuration_utils.py", line 662, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/configuration_utils.py", line 721, in _get_config_dict
    resolved_config_file = cached_file(
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/hub.py", line 322, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/transformers/utils/hub.py", line 553, in cached_files
    raise OSError(
OSError: We couldn't connect to 'https://huggingface.co ' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode '.