# default
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=16   | tee  tf_cnn_benchmarks16.log
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=32   | tee  tf_cnn_benchmarks32.log
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=64   | tee  tf_cnn_benchmarks64.log
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=128   | tee  tf_cnn_benchmarks128.log
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=256   | tee  tf_cnn_benchmarks256.log
#--use_fp16
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=16  --use_fp16 | tee  tf_cnn_benchmarks16_fp16.log
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=32  --use_fp16 | tee  tf_cnn_benchmarks32_fp16.log
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=64  --use_fp16 | tee  tf_cnn_benchmarks64_fp16.log
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=128  --use_fp16 | tee  tf_cnn_benchmarks128_fp16.log
python3 /benchmarks/tensorflow_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py  --batch_size=256  --use_fp16 | tee  tf_cnn_benchmarks256_fp16.log

