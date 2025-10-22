rm -f *.token *.sig *.manifest.sgx *.manifest msg_pb2.py msg_pb2_grpc.py
/data1/junjie_chen/.conda/envs/jjchen/bin/python3.10 -m grpc_tools.protoc \
        -I /home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/ \
        --python_out=/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/ \
        --grpc_python_out=/home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu/ \
        /home/fdcffcf0-4e53-40aa-a255-19c2675ad6b1/TSQP/tee_gpu//msg.proto
gramine-manifest \
        -Dlog_level=error \
        -Darch_libdir=/lib/x86_64-linux-gnu \
        -Dentrypoint=/data1/junjie_chen/.conda/envs/jjchen/bin/python3.10 \
        --template tee_runner.manifest.template > tee_runner.manifest
Usage: gramine-manifest [OPTIONS] [INFILE] [OUTFILE]
Try 'gramine-manifest --help' for help.

Error: No such option: --template
make: *** [Makefile:41: tee_runner.manifest] Error 2