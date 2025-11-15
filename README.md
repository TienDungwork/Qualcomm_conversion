# Qualcomm AI Model Conversion Workflow

## 1. Tải SDK trên Ubuntu 22.04 (bắt buộc 22.04)

Download SDK từ: https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK

```bash
unzip v2.40.0.251030.zip
cd qairt/v2.40.0.251030
export QAIRT_ROOT=`pwd`
```

## 2. Thiết lập môi trường

```bash
sudo bash bin/check-linux-dependency.sh
python3.10 -m venv "<venv_path>"
source <venv_path>/bin/activate
bin/check-python-dependency
pip install -r requirements.txt
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
source bin/envsetup.sh
```

## 3. Các lệnh chuyển đổi model ONNX sang QNN binary

### Mẫu (YOLOv8)

```bash
python convert.py -m <path_to_onnx> -i images -s 1,3,640,640
```

### Các tham số

**Bắt buộc:**
- `-m`: Đường dẫn file ONNX
- `-i`: Tên input tensor (thường là `images` với YOLO)
- `-s`: Shape input (batch,channels,height,width)

**Tùy chọn:**
- `-o`: Tên file binary output (không cần .bin)
- `-d`: Thư mục lưu file binary (mặc định: `compiled_models`)
- `-p 16`: Float 16-bit (nhẹ hơn, nhanh hơn)
- `-p 32`: Float 32-bit (chính xác hơn, nặng hơn)
- `-c`: Đường dẫn file config JSON cho HTP backend
- `--dlc-only`: Dừng sau bước convert ONNX → DLC (không tạo binary)

### Ví dụ

```bash
python convert_model.py \
    -m yolov8n.onnx \
    -i images \
    -s 1,3,640,640 \
    -p 16 \
    -d compiled_models \
    -o yolov8n_fp16 \
    -c HtpConfigFile.json
```
