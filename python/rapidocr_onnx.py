#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RapidOcrOnnx Python API
基于OcrLiteCApi.h创建的Python绑定
"""

import os
import ctypes
import platform
from typing import List, Tuple, Optional, Dict, Any

# 定义结构体类
class OCR_PARAM(ctypes.Structure):
    _fields_ = [
        ("padding", ctypes.c_int),
        ("maxSideLen", ctypes.c_int),
        ("boxScoreThresh", ctypes.c_float),
        ("boxThresh", ctypes.c_float),
        ("unClipRatio", ctypes.c_float),
        ("doAngle", ctypes.c_int),  # 1 means do
        ("mostAngle", ctypes.c_int),  # 1 means true
    ]

class OCR_POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
    ]

class OCR_INPUT(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("type", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("dataLength", ctypes.c_long),
    ]

class TEXT_BLOCK(ctypes.Structure):
    pass

TEXT_BLOCK._fields_ = [
    ("boxPoint", ctypes.POINTER(OCR_POINT)),
    ("boxScore", ctypes.c_float),
    ("angleIndex", ctypes.c_int),
    ("angleScore", ctypes.c_float),
    ("angleTime", ctypes.c_double),
    ("text", ctypes.POINTER(ctypes.c_uint8)),
    ("charScores", ctypes.POINTER(ctypes.c_float)),
    ("charScoresLength", ctypes.c_ulonglong),
    ("boxPointLength", ctypes.c_ulonglong),
    ("textLength", ctypes.c_ulonglong),
    ("crnnTime", ctypes.c_double),
    ("blockTime", ctypes.c_double),
]

class OCR_RESULT(ctypes.Structure):
    _fields_ = [
        ("dbNetTime", ctypes.c_double),
        ("textBlocks", ctypes.POINTER(TEXT_BLOCK)),
        ("textBlocksLength", ctypes.c_ulonglong),
        ("detectTime", ctypes.c_double),
    ]

# 加载动态库
def _load_library() -> ctypes.CDLL:
    """
    根据操作系统加载相应的动态库
    """
    system = platform.system()
    lib_name = ""
    
    if system == "Windows":
        lib_name = "OcrLiteOnnx.dll"
    elif system == "Linux":
        lib_name = "libRapidOcrOnnx.so"
    elif system == "Darwin":  # macOS
        lib_name = "libOcrLiteOnnx.dylib"
    else:
        raise NotImplementedError(f"Unsupported operating system: {system}")
    
    # 尝试在多个路径查找库文件
    lib_paths = [
        os.path.join(os.path.dirname(__file__), lib_name),
        os.path.join(os.path.dirname(__file__), "build", "Release", lib_name),
        os.path.join(os.path.dirname(__file__), "build", "Debug", lib_name),
        os.path.join(os.path.dirname(__file__), "build", lib_name),
        # 添加库的默认构建路径
        os.path.join(os.path.dirname(__file__), "lib", lib_name),
        # 添加可能的系统路径
        os.path.join("/usr/local/lib", lib_name),
        lib_name  # 系统路径
    ]
    
    for path in lib_paths:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except Exception as e:
                print(f"Failed to load library from {path}: {e}")
    
    raise FileNotFoundError(f"Could not find {lib_name} in any of the search paths")

# 加载库并定义函数接口
try:
    _lib = _load_library()
    
    # 定义函数接口
    _lib.OcrInit.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int
    ]
    _lib.OcrInit.restype = ctypes.c_void_p
    
    _lib.OcrDetect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(OCR_PARAM)
    ]
    _lib.OcrDetect.restype = ctypes.c_char
    
    _lib.OcrDetectInput.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(OCR_INPUT),
        ctypes.POINTER(OCR_PARAM),
        ctypes.POINTER(OCR_RESULT),
        ctypes.c_bool
    ]
    _lib.OcrDetectInput.restype = ctypes.c_char
    
    _lib.OcrFreeResult.argtypes = [ctypes.POINTER(OCR_RESULT)]
    _lib.OcrFreeResult.restype = ctypes.c_char
    
    _lib.OcrGetLen.argtypes = [ctypes.c_void_p]
    _lib.OcrGetLen.restype = ctypes.c_int
    
    _lib.OcrGetResult.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int
    ]
    _lib.OcrGetResult.restype = ctypes.c_char
    
    _lib.OcrDestroy.argtypes = [ctypes.c_void_p]
    _lib.OcrDestroy.restype = None
except Exception as e:
    print(f"Warning: Failed to load OCR library: {e}")
    _lib = None

class RapidOCR:
    """
    RapidOcrOnnx Python API 封装类
    """
    
    def __init__(self, 
                 det_model_path: str, 
                 cls_model_path: str, 
                 rec_model_path: str, 
                 keys_path: Optional[str] = None, 
                 n_threads: int = 4):
        """
        初始化OCR引擎
        
        Args:
            det_model_path: 检测模型路径
            cls_model_path: 角度分类模型路径
            rec_model_path: 识别模型路径
            keys_path: 字符集路径，可选
            n_threads: 线程数
        """
        if _lib is None:
            raise RuntimeError("OCR library not loaded properly")
        
        # 确保路径是绝对路径并转换为字节
        det_model = os.path.abspath(det_model_path).encode('utf-8')
        cls_model = os.path.abspath(cls_model_path).encode('utf-8')
        rec_model = os.path.abspath(rec_model_path).encode('utf-8')
        keys = os.path.abspath(keys_path).encode('utf-8') if keys_path else None
        
        self._handle = _lib.OcrInit(det_model, cls_model, rec_model, keys, n_threads)
        if not self._handle:
            raise RuntimeError("Failed to initialize OCR engine")
        
    def __del__(self):
        """
        释放OCR引擎资源
        """
        if hasattr(self, '_handle') and self._handle and _lib:
            _lib.OcrDestroy(self._handle)
            self._handle = None
    
    def detect(self, 
               img_path: str, 
               img_name: str = "",
               padding: int = 50,
               max_side_len: int = 1024,
               box_score_thresh: float = 0.6,
               box_thresh: float = 0.3,
               un_clip_ratio: float = 2.0,
               do_angle: bool = True,
               most_angle: bool = True) -> str:
        """
        从图像文件中检测文字
        
        Args:
            img_path: 图像路径
            img_name: 图像名称（可选）
            padding: 图像边缘填充
            max_side_len: 图像最大边长
            box_score_thresh: 文本框得分阈值
            box_thresh: 文本框阈值
            un_clip_ratio: 文本框裁剪参数
            do_angle: 是否进行方向检测
            most_angle: 是否使用多数角度
            
        Returns:
            识别结果文本
        """
        # 设置参数
        param = OCR_PARAM()
        param.padding = padding
        param.maxSideLen = max_side_len
        param.boxScoreThresh = box_score_thresh
        param.boxThresh = box_thresh
        param.unClipRatio = un_clip_ratio
        param.doAngle = 1 if do_angle else 0
        param.mostAngle = 1 if most_angle else 0
        
        # 转换路径为字节
        img_path_bytes = os.path.abspath(img_path).encode('utf-8')
        img_name_bytes = img_name.encode('utf-8') if img_name else b""
        
        # 执行检测
        success = _lib.OcrDetect(self._handle, img_path_bytes, img_name_bytes, ctypes.byref(param))
        if not success:
            raise RuntimeError("OCR detection failed")
        
        # 获取结果
        result_len = _lib.OcrGetLen(self._handle)
        if result_len <= 1:  # 至少有一个结束符
            return ""
        
        # 分配缓冲区并获取结果
        buffer = ctypes.create_string_buffer(result_len)
        _lib.OcrGetResult(self._handle, buffer, result_len)
        
        return buffer.value.decode('utf-8', errors='ignore')
    
    def detect_image_bytes(self, 
                           image_bytes: bytes,
                           is_gray: bool = False,
                           padding: int = 0,
                           max_side_len: int = 2000,
                           box_score_thresh: float = 0.3,
                           box_thresh: float = 0.5,
                           un_clip_ratio: float = 1.6,
                           do_angle: bool = False,
                           most_angle: bool = False,
                           is_recog: bool = False) -> List[Dict[str, Any]]:
        """
        从图像字节数据中检测文字
        
        Args:
            image_bytes: 图像字节数据
            is_gray: 是否为灰度图像
            padding: 图像边缘填充
            max_side_len: 图像最大边长
            box_score_thresh: 文本框得分阈值
            box_thresh: 文本框阈值
            un_clip_ratio: 文本框裁剪参数
            do_angle: 是否进行方向检测
            most_angle: 是否使用多数角度
            is_recog: 是否进行文字识别
            
        Returns:
            识别结果列表，每个元素包含文本和坐标等信息
        """
        # 设置输入参数
        input_data = OCR_INPUT()
        input_data.data = ctypes.cast(ctypes.create_string_buffer(image_bytes), ctypes.POINTER(ctypes.c_uint8))
        input_data.type = 1  # 1表示图像字节数据
        input_data.channels = 1 if is_gray else 3
        input_data.width = 0  # 对于字节数据，这些参数不重要
        input_data.height = 0
        input_data.dataLength = len(image_bytes)
        
        # 设置OCR参数
        param = OCR_PARAM()
        param.padding = padding
        param.maxSideLen = max_side_len
        param.boxScoreThresh = box_score_thresh
        param.boxThresh = box_thresh
        param.unClipRatio = un_clip_ratio
        param.doAngle = 1 if do_angle else 0
        param.mostAngle = 1 if most_angle else 0
        
        # 准备结果结构
        result = OCR_RESULT()
        
        # 执行检测
        success = _lib.OcrDetectInput(self._handle, ctypes.byref(input_data), ctypes.byref(param), ctypes.byref(result), is_recog)
        
        # 处理结果
        results = []
        if success and result.textBlocks and result.textBlocksLength > 0:
            for i in range(result.textBlocksLength):
                text_block = result.textBlocks[i]
                
                # 提取文本
                text_bytes = bytes(ctypes.string_at(text_block.text, text_block.textLength))
                text = text_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
                
                # 提取坐标点
                points = []
                if text_block.boxPoint and text_block.boxPointLength > 0:
                    for j in range(text_block.boxPointLength):
                        points.append((text_block.boxPoint[j].x, text_block.boxPoint[j].y))
                
                # 提取字符得分
                char_scores = []
                if text_block.charScores and text_block.charScoresLength > 0:
                    char_scores = [text_block.charScores[j] for j in range(text_block.charScoresLength)]
                
                results.append({
                    'text': text,
                    'box_score': text_block.boxScore,
                    'angle_index': text_block.angleIndex,
                    'angle_score': text_block.angleScore,
                    'box_points': points,
                    'char_scores': char_scores,
                    'crnn_time': text_block.crnnTime,
                    'block_time': text_block.blockTime
                })
        
        # 释放结果内存
        _lib.OcrFreeResult(ctypes.byref(result))
        
        return results
