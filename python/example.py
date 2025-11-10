from rapidocr_onnx import RapidOCR

# 示例用法
def main():
    """
    示例：如何使用RapidOCR类
    """
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path> [det_model_path] [cls_model_path] [rec_model_path] [keys_path]")
        return
    
    image_path = sys.argv[1]
    
    # 默认模型路径
    det_model = "../external/RapidOcrOnnxLibTest/resource/models/ch_PP-OCRv4_det_infer.onnx"
    cls_model = "../external/RapidOcrOnnxLibTest/resource/models/ch_ppocr_mobile_v2.0_cls_infer.onnx"
    #rec_model = "../external/models/ch_PP-OCRv3_rec_infer.onnx"
    rec_model = "../external/RapidOcrOnnxLibTest/resource/models/rec.bin"
    keys_path = "../external/RapidOcrOnnxLibTest/resource/models/ppocr_keys_v1.txt" 
    
    # 从命令行参数覆盖默认路径
    if len(sys.argv) > 2:
        det_model = sys.argv[2]
    if len(sys.argv) > 3:
        cls_model = sys.argv[3]
    if len(sys.argv) > 4:
        rec_model = sys.argv[4]
    if len(sys.argv) > 5:
        keys_path = sys.argv[5]
    
    try:
        # 初始化OCR引擎
        ocr = RapidOCR(det_model, cls_model, rec_model, n_threads=4)
        
        # 方式1：使用文件路径检测
        print("===== 检测结果（文件路径）=====")
        result = ocr.detect(image_path)
        print(result)
        
        # 方式2：使用图像字节数据检测（更详细的结果）
        print("\n===== 检测结果（字节数据）=====")
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        detailed_results = ocr.detect_image_bytes(img_bytes)
        
        for i, res in enumerate(detailed_results):
            print(f"\n文本块 {i+1}:")
            print(f"  文本: {res['text']}")
            print(f"  置信度: {res['box_score']:.4f}")
            print(f"  坐标: {res['box_points']}")
            
    except Exception as e:
        print(f"Error: {e}")

#python rapidocr_onnx.py /data/github/RapidOcrOnnxLibTest/resource/images/0.84691966.1RuxaTk9Z4GWyU-mIDcw5w\=\=.jpg
if __name__ == "__main__":
    main()