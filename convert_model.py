#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

QNN_SDK_ROOT = "/home/atin/ntiendung/qairt/qairt/2.40.0.251030"

def setup_environment():
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{QNN_SDK_ROOT}/lib/python:{env.get('PYTHONPATH', '')}"
    env['LD_LIBRARY_PATH'] = f"{QNN_SDK_ROOT}/lib/x86_64-linux-clang:{env.get('LD_LIBRARY_PATH', '')}"
    env['PATH'] = f"{QNN_SDK_ROOT}/bin/x86_64-linux-clang:{env.get('PATH', '')}"
    env['QNN_SDK_ROOT'] = QNN_SDK_ROOT
    return env

def convert_onnx_to_dlc(onnx_path, input_name, input_shape, float_precision=16, preserve_io=True, env=None):
    onnx_path = Path(onnx_path)
    dlc_path = onnx_path.with_suffix('.dlc')
    
    cmd = [
        f"{QNN_SDK_ROOT}/bin/x86_64-linux-clang/qairt-converter",
        "--input_network", str(onnx_path),
        "--output_path", str(dlc_path),
        "--source_model_input_shape", input_name, input_shape,
        "--float_bitwidth", str(float_precision),
        "--float_bias_bitwidth", str(float_precision),
    ]
    
    if preserve_io:
        cmd.append("--preserve_io")
    
    print(f"Converting {onnx_path} to DLC...")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    output, error = proc.communicate()
    
    print(output.decode(), error.decode())
    
    if proc.returncode != 0:
        return None
    
    return dlc_path

def generate_context_binary(dlc_path, output_name=None, config_file=None, output_dir=None, log_level="info", env=None):
    dlc_path = Path(dlc_path)
    
    if output_name is None:
        output_name = dlc_path.stem
    
    if output_dir is None:
        output_dir = "compiled_models"
    
    if config_file is None:
        config_file = "/home/atin/ntiendung/qairt/qairt/scripts/HtpConfigFile.json"
    
    Path(output_dir).mkdir(exist_ok=True)
    
    cmd = [
        f"{QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-context-binary-generator",
        "--dlc_path", str(dlc_path),
        "--backend", f"{QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so",
        "--model", f"{QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnModelDlc.so",
        "--binary_file", output_name,
        "--output_dir", output_dir,
        f"--log_level={log_level}"
    ]
    
    if Path(config_file).exists():
        cmd.extend(["--config_file", config_file])
    
    print(f"Generating binary for {dlc_path}...")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    output, error = proc.communicate()
    
    print(output.decode(), error.decode())
    
    if proc.returncode != 0:
        return None
    
    binary_path = Path(output_dir) / f"{output_name}.bin"
    return binary_path if binary_path.exists() else None

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to QNN binary')
    parser.add_argument('--model', '-m', required=True, help='ONNX model path')
    parser.add_argument('--input_name', '-i', required=True, help='Input tensor name')
    parser.add_argument('--input_shape', '-s', required=True, help='Input shape (e.g., 1,3,640,640)')
    parser.add_argument('--output_name', '-o', help='Output binary name')
    parser.add_argument('--output_dir', '-d', default='compiled_models', help='Output directory')
    parser.add_argument('--precision', '-p', type=int, choices=[16, 32], default=16, help='Float precision')
    parser.add_argument('--config', '-c', help='HTP config file')
    parser.add_argument('--log_level', '-l', default='info', choices=['verbose', 'info', 'warn', 'error'], help='Log level')
    parser.add_argument('--no-preserve-io', action='store_true', help='Do not preserve I/O shapes')
    parser.add_argument('--dlc-only', action='store_true', help='Only convert to DLC')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: {args.model} not found")
        sys.exit(1)
    
    env = setup_environment()
    
    dlc_path = convert_onnx_to_dlc(
        onnx_path=args.model,
        input_name=args.input_name,
        input_shape=args.input_shape,
        float_precision=args.precision,
        preserve_io=not args.no_preserve_io,
        env=env
    )
    
    if dlc_path is None:
        sys.exit(1)
    
    if args.dlc_only:
        print(f"DLC: {dlc_path}")
        sys.exit(0)
    
    binary_path = generate_context_binary(
        dlc_path=dlc_path,
        output_name=args.output_name,
        config_file=args.config,
        output_dir=args.output_dir,
        log_level=args.log_level,
        env=env
    )
    
    if binary_path is None:
        sys.exit(1)
    
    print(f"Binary: {binary_path}")

if __name__ == "__main__":
    main()
