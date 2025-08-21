#include "OcrLiteImpl.h"
#include "OcrUtils.h"
#include <stdarg.h> //windows&linux

OcrLiteImpl::OcrLiteImpl() {
    loggerBuffer = (char *)malloc(8192);
}

OcrLiteImpl::~OcrLiteImpl() {
    if (isOutputResultTxt) {
        fclose(resultTxt);
    }
    free(loggerBuffer);
}

void OcrLiteImpl::setNumThread(int numOfThread) {
    dbNet.setNumThread(numOfThread);
    angleNet.setNumThread(numOfThread);
    crnnNet.setNumThread(numOfThread);
}

void OcrLiteImpl::initLogger(bool isConsole, bool isPartImg, bool isResultImg) {
    isOutputConsole = isConsole;
    isOutputPartImg = isPartImg;
    isOutputResultImg = isResultImg;
    printf("init console:%d partimg:%d resultimg:%d\n", isOutputConsole, isOutputPartImg, isOutputResultImg);
}

void OcrLiteImpl::enableResultTxt(const char *path, const char *imgName) {
    isOutputResultTxt = true;
    std::string resultTxtPath = getResultTxtFilePath(path, imgName);
    printf("resultTxtPath(%s)\n", resultTxtPath.c_str());
    resultTxt = fopen(resultTxtPath.c_str(), "w");
}

void OcrLiteImpl::setGpuIndex(int gpuIndex) {
    dbNet.setGpuIndex(gpuIndex);
    angleNet.setGpuIndex(gpuIndex);
    crnnNet.setGpuIndex(gpuIndex);
}

bool OcrLiteImpl::initModels(const std::string &detPath, const std::string &clsPath,
                         const std::string &recPath) {
    Logger("=====Init Models=====\n");
    Logger("--- Init DbNet ---\n");
    dbNet.initModel(detPath);

    Logger("--- Init AngleNet ---\n");
    angleNet.initModel(clsPath);

    Logger("--- Init CrnnNet ---\n");
    crnnNet.initModel(recPath);

    Logger("Init Models Success!\n");
    return true;
}

bool OcrLiteImpl::initModels(const std::string &detPath, const std::string &clsPath,
                         const std::string &recPath, const std::string &keysPath) {
    Logger("=====Init Models=====\n");
    Logger("--- Init DbNet ---\n");
    dbNet.initModel(detPath);

    Logger("--- Init AngleNet ---\n");
    angleNet.initModel(clsPath);

    Logger("--- Init CrnnNet ---\n");
    crnnNet.initModel(recPath, keysPath);

    Logger("Init Models Success!\n");
    return true;
}

void OcrLiteImpl::Logger(const char *format, ...) {
    if (!(isOutputConsole || isOutputResultTxt)) return;
    memset(loggerBuffer, 0, 8192);
    va_list args;
    va_start(args, format);
    vsprintf(loggerBuffer, format, args);
    va_end(args);
    if (isOutputConsole) printf("%s", loggerBuffer);
    if (isOutputResultTxt) fprintf(resultTxt, "%s", loggerBuffer);
}

cv::Mat makePadding(cv::Mat &src, const int padding) {
    if (padding <= 0) return src;
    cv::Scalar paddingScalar = {255, 255, 255};
    cv::Mat paddingSrc;
    cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, cv::BORDER_ISOLATED, paddingScalar);
    return paddingSrc;
}

cv::Mat crop_and_flip(const cv::Mat& image, int h_ratio = 8, int w_ratio = 2) {
    // 获取图像高度和宽度
    int height = image.rows;
    int width = image.cols;

    // 计算裁剪尺寸
    int crop_height = height / h_ratio;
    int crop_width = width / w_ratio;

    // 裁剪左下角区域
    cv::Rect left_bottom_roi(0, height - crop_height, crop_width, crop_height);
    cv::Mat left_bottom_crop = image(left_bottom_roi);

    // 裁剪右下角区域
    cv::Rect right_bottom_roi(width - crop_width, height - crop_height, crop_width, crop_height);
    cv::Mat right_bottom_crop = image(right_bottom_roi);

    // 水平翻转两个裁剪图像
    cv::Mat left_bottom_flip, right_bottom_flip;
    cv::flip(left_bottom_crop, left_bottom_flip, 1);  // 1 = 水平翻转
    cv::flip(right_bottom_crop, right_bottom_flip, 1);

    // 水平拼接原图与翻转图
    cv::Mat left_combined, right_combined;
    cv::hconcat(left_bottom_crop, left_bottom_flip, left_combined);
    cv::hconcat(right_bottom_crop, right_bottom_flip, right_combined);

    // 再将两个结果图像水平拼接
    cv::Mat result_image;
    cv::hconcat(left_combined, right_combined, result_image);

    return result_image;
}

OcrResult OcrLiteImpl::detect(const char *path, const char *imgName,
                          const int padding, const int maxSideLen,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle, bool isRecog) {
    std::string imgFile = getSrcImgFilePath(path, imgName);
    cv::Mat originSrc = imread(imgFile, cv::IMREAD_COLOR);//default : BGR

    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    resize += 2 * padding;
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    OcrResult result;
    result = detect(path, imgName, paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle, isRecog);
    return result;
}


OcrResult OcrLiteImpl::detectImageBytes(const uint8_t *data, const long dataLength, const int grey,
                                    const int padding, const int maxSideLen,
                                    float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle,
                                    bool mostAngle, bool isRecog) {
    std::vector<uint8_t> vecData(data, data + dataLength);
    cv::Mat originSrc = cv::imdecode(vecData, grey == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);//default : BGR
    OcrResult result;
    result = detect(originSrc, padding, maxSideLen,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle, isRecog);
    return result;

}

OcrResult OcrLiteImpl::detectBitmap(uint8_t *bitmapData, int width, int height, int channels, int padding,
                                int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle,
                                bool mostAngle, bool isRecog) {
    cv::Mat originSrc(height, width, CV_8UC(channels), bitmapData);
    if (channels > 3) {
        cv::cvtColor(originSrc, originSrc, cv::COLOR_RGBA2BGR);
    } else if (channels == 3) {
        cv::cvtColor(originSrc, originSrc, cv::COLOR_RGB2BGR);
    }
    //
    if(isRecog){
        originSrc = crop_and_flip(originSrc);
        //cv::imwrite("crop_and_flip.png", originSrc);
    }

    OcrResult result;
    result = detect(originSrc, padding, maxSideLen,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle, isRecog);
    return result;
}


OcrResult OcrLiteImpl::detect(const cv::Mat &mat, int padding, int maxSideLen, float boxScoreThresh, float boxThresh,
                          float unClipRatio, bool doAngle, bool mostAngle, bool isRecog) {
    cv::Mat originSrc = mat;
    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    resize += 2 * padding;
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    OcrResult result;
    std::string path = "", imgName = "";
    fprintf(stderr, "path:%s - imgName:%s\n", path.c_str(), imgName.c_str()); fflush(stderr);
    result = detect(path.c_str(), imgName.c_str(), paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle, isRecog);
    return result;
}

std::vector<cv::Mat> OcrLiteImpl::getPartImages(cv::Mat &src, std::vector<TextBox> &textBoxes,
                                            const char *path, const char *imgName) {
    std::vector<cv::Mat> partImages;
    for (size_t i = 0; i < textBoxes.size(); ++i) {
        cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
        partImages.emplace_back(partImg);
        //OutPut DebugImg
        if (isOutputPartImg) {
            std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-part-");
            fprintf(stderr, "debugImgFile:%s\n", debugImgFile.c_str()); fflush(stderr);
            saveImg(partImg, debugImgFile.c_str());
        }
    }
    return partImages;
}

OcrResult OcrLiteImpl::detect(const char *path, const char *imgName,
                          cv::Mat &src, cv::Rect &originRect, ScaleParam &scale,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle, bool isRecog) {

    cv::Mat textBoxPaddingImg = src.clone();
    int thickness = getThickness(src);

    Logger("=====Start detect=====\n");
    Logger("ScaleParam(sw:%d,sh:%d,dw:%d,dh:%d,%f,%f)\n", scale.srcWidth, scale.srcHeight,
           scale.dstWidth, scale.dstHeight,
           scale.ratioWidth, scale.ratioHeight);

    Logger("---------- step: dbNet getTextBoxes ----------\n");
    double startTime = getCurrentTime();
    std::vector<TextBox> textBoxes = dbNet.getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
    double endDbNetTime = getCurrentTime();
    double dbNetTime = endDbNetTime - startTime;
    Logger("dbNetTime(%fms)\n", dbNetTime);

    for (size_t i = 0; i < textBoxes.size(); ++i) {
        Logger("TextBox[%d](+padding)[score(%f),[x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d]]\n", i,
               textBoxes[i].score,
               textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[0].y,
               textBoxes[i].boxPoint[1].x, textBoxes[i].boxPoint[1].y,
               textBoxes[i].boxPoint[2].x, textBoxes[i].boxPoint[2].y,
               textBoxes[i].boxPoint[3].x, textBoxes[i].boxPoint[3].y);
    }

    Logger("---------- step: drawTextBoxes ----------\n");
    drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

    //---------- getPartImages ----------
    fprintf(stderr, "path:%s imgName:%s\n", path, imgName); fflush(stderr);
    std::vector<cv::Mat> partImages = getPartImages(src, textBoxes, path, imgName);

    Logger("---------- step: angleNet getAngles ----------\n");
    std::vector<Angle> angles;
    angles = angleNet.getAngles(partImages, path, imgName, doAngle, mostAngle);

    //Log Angles
    for (size_t i = 0; i < angles.size(); ++i) {
        Logger("angle[%d][index(%d), score(%f), time(%fms)]\n", i, angles[i].index, angles[i].score, angles[i].time);
    }

    //Rotate partImgs
    for (size_t i = 0; i < partImages.size(); ++i) {
        if (angles[i].index == 1) {
            partImages.at(i) = matRotateClockWise180(partImages[i]);
        }
    }

    std::vector<TextBlock> textBlocks;
    std::vector<TextLine> textLines;
    if(isRecog){
        Logger("---------- step: crnnNet getTextLine ----------\n");
        textLines = crnnNet.getTextLines(partImages, path, imgName);
        //Log TextLines
        for (size_t i = 0; i < textLines.size(); ++i) {
            Logger("textLine[%d](%s)\n", i, textLines[i].text.c_str());
            std::ostringstream txtScores;
            for (size_t s = 0; s < textLines[i].charScores.size(); ++s) {
                if (s == 0) {
                    txtScores << textLines[i].charScores[s];
                } else {
                    txtScores << " ," << textLines[i].charScores[s];
                }
            }
            Logger("textScores[%d]{%s}\n", i, std::string(txtScores.str()).c_str());
            Logger("crnnTime[%d](%fms)\n", i, textLines[i].time);
        }

        
    }
    for (size_t i = 0; i < textBoxes.size(); ++i) {
        std::vector<cv::Point> boxPoint = std::vector<cv::Point>(4);
        int padding = originRect.x;//padding conversion
        boxPoint[0] = cv::Point(textBoxes[i].boxPoint[0].x - padding, textBoxes[i].boxPoint[0].y - padding);
        boxPoint[1] = cv::Point(textBoxes[i].boxPoint[1].x - padding, textBoxes[i].boxPoint[1].y - padding);
        boxPoint[2] = cv::Point(textBoxes[i].boxPoint[2].x - padding, textBoxes[i].boxPoint[2].y - padding);
        boxPoint[3] = cv::Point(textBoxes[i].boxPoint[3].x - padding, textBoxes[i].boxPoint[3].y - padding);
        std::string textlines_text_proxy = isRecog ? textLines[i].text : ""; 
        std::vector<float> charscore_proxy = isRecog ? textLines[i].charScores : std::vector<float>(); 
        double txtlines_time_proxy = isRecog ? textLines[i].time : 0.;
        TextBlock textBlock{boxPoint, textBoxes[i].score, angles[i].index, angles[i].score,
                            angles[i].time, textlines_text_proxy, charscore_proxy, txtlines_time_proxy,
                            angles[i].time + txtlines_time_proxy};
        textBlocks.emplace_back(textBlock);
    }


    double endTime = getCurrentTime();
    double fullTime = endTime - startTime;
    Logger("=====End detect=====\n");
    Logger("FullDetectTime(%fms)\n", fullTime);

    //cropped to original size
    cv::Mat textBoxImg;

    if (originRect.x > 0 && originRect.y > 0) {
        textBoxPaddingImg(originRect).copyTo(textBoxImg);
    } else {
        textBoxImg = textBoxPaddingImg;
    }

    //Save result.jpg
    if (isOutputResultImg) {
        fprintf(stderr, "path %s\n", path); fflush(stderr);
        std::string resultImgFile = getResultImgFilePath(path, imgName);
        fprintf(stderr, "save %s\n", resultImgFile.c_str()); fflush(stderr);
        imwrite(resultImgFile, textBoxImg);
    }

    std::string strRes;
    for (auto &textBlock: textBlocks) {
        strRes.append(textBlock.text);
        strRes.append("\n");
    }

    return OcrResult{dbNetTime, textBlocks, textBoxImg, fullTime, strRes};
}
