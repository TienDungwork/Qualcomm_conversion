import os
import subprocess

QNN_SDK_ROOT = "/home/atin/ntiendung/qairt/qairt/2.40.0.251030"

# Setup environment variables for QNN SDK
env = os.environ.copy()
env['PYTHONPATH'] = f"{QNN_SDK_ROOT}/lib/python:{env.get('PYTHONPATH', '')}"
env['LD_LIBRARY_PATH'] = f"{QNN_SDK_ROOT}/lib/x86_64-linux-clang:{env.get('LD_LIBRARY_PATH', '')}"
env['PATH'] = f"{QNN_SDK_ROOT}/bin/x86_64-linux-clang:{env.get('PATH', '')}"
env['QNN_SDK_ROOT'] = QNN_SDK_ROOT

 
proc = subprocess.Popen([QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qairt-converter",
                        "--input_network", "/home/atin/ntiendung/qairt/qairt/scripts/test_yolov9/yolov8n.onnx",
                        "--output_path", "/home/atin/ntiendung/qairt/qairt/scripts/test_yolov9/yolov8n.dlc",
                        "--input_dim", "images", "1,3,640,640",
                        "--float_bitwidth", "16",
                        "--float_bias_bitwidth", "16",
                        "--preserve_io",
                        "--onnx_define_symbol", "sequence_len", "1"
                       ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

output, error = proc.communicate()

print(output.decode(),error.decode())


proc = subprocess.Popen([QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qnn-context-binary-generator",
                        "--dlc_path", "last_17h16_24_09_yolov11_detect_float.dlc",
                        "--backend", QNN_SDK_ROOT + "/lib/x86_64-linux-clang/libQnnHtp.so",
                        "--model", QNN_SDK_ROOT + "/lib/x86_64-linux-clang/libQnnModelDlc.so",
                        "--config_file", r"/home/atin/ntiendung/qairt/qairt/scripts/HtpConfigFile.json",
                        "--binary_file", "yolov8n",
                        "--output_dir", "compiled_output_dir",
                        "--log_level=verbose"
                        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

output, error = proc.communicate()

print(output.decode(), error.decode())