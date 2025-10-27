Traceback (most recent call last):
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/server_optimized.py", line 301, in <module>
    main()
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/server_optimized.py", line 297, in main
    server.serve()
  File "/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/server_optimized.py", line 252, in serve
    response_bytes = msgpack.packb(response, use_bin_type=True)
  File "/data1/junjie_chen/.conda/envs/jjchen/lib/python3.10/site-packages/msgpack/__init__.py", line 36, in packb
    return Packer(**kwargs).pack(o)
  File "msgpack/_packer.pyx", line 279, in msgpack._cmsgpack.Packer.pack
  File "msgpack/_packer.pyx", line 276, in msgpack._cmsgpack.Packer.pack
  File "msgpack/_packer.pyx", line 270, in msgpack._cmsgpack.Packer._pack
  File "msgpack/_packer.pyx", line 213, in msgpack._cmsgpack.Packer._pack_inner
  File "msgpack/_packer.pyx", line 270, in msgpack._cmsgpack.Packer._pack
  File "msgpack/_packer.pyx", line 213, in msgpack._cmsgpack.Packer._pack_inner
  File "msgpack/_packer.pyx", line 270, in msgpack._cmsgpack.Packer._pack
  File "msgpack/_packer.pyx", line 213, in msgpack._cmsgpack.Packer._pack_inner
  File "msgpack/_packer.pyx", line 270, in msgpack._cmsgp