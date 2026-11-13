#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// ---------- 小工具：中位数 ----------
static double medianOf(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------- 掩膜质量评估：判断它是"回形针"还是"背景" ----------
struct MaskEval
{
    bool   valid    = false;  // 是否是一个说得通的物体掩膜
    double solidity = 1.0;    // 面积加权平均实心度，越小越像细丝环
    double fgRatio  = 0.0;    // 前景占整图比例
    int    comps    = 0;      // 有效连通域个数
};

static MaskEval evaluateMask(const cv::Mat& mask, double minCompArea)
{
    MaskEval e;
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    const double total = (double)cv::countNonZero(mask);
    if (total <= 0.0) return e;
    e.fgRatio = total / (double)mask.total();

    double largest = 0.0, solidSum = 0.0, wSum = 0.0;
    for (int i = 1; i < n; ++i)
    {
        const double a = (double)stats.at<int>(i, cv::CC_STAT_AREA);
        if (a < minCompArea) continue;      // 太小的当噪声，不参与判定
        ++e.comps;
        largest = std::max(largest, a);

        std::vector<std::vector<cv::Point>> cs;
        cv::findContours(labels == i, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (cs.empty()) continue;

        // 实心度 = 轮廓面积 / 凸包面积：细丝绕成的环会明显小于 1
        std::vector<cv::Point> hull;
        cv::convexHull(cs[0], hull);
        const double hullArea = cv::contourArea(hull);
        const double solidity = (hullArea > 1.0) ? std::min(1.0, a / hullArea) : 1.0;
        solidSum += solidity * a;
        wSum += a;
    }

    if (e.comps < 2 || wSum <= 0.0) return e;
    e.solidity = solidSum / wSum;

    // 背景伪解的特征：一个大连通域吃掉了绝大部分前景
    if (largest / total > 0.80 && e.fgRatio > 0.20) return e;

    e.valid = (e.fgRatio > 0.0015 && e.fgRatio < 0.65);
    return e;
}

int main(int argc, char** argv)
{
    const std::string imagePath = (argc > 1) ? argv[1] : "1.png";
    const std::string savePath  = "count_result.png";

    cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cerr << "无法读取图片: " << imagePath << std::endl;
        return -1;
    }

    // ---------- 1. 灰度 + 尺度归一（只缩不放，避免放大引入伪影） ----------
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    const double kTargetLongSide = 900.0;
    double scale = 1.0;
    const double longSide = (double)std::max(gray.cols, gray.rows);
    if (longSide > kTargetLongSide) scale = kTargetLongSide / longSide;

    cv::Mat work;
    if (scale < 1.0)
        cv::resize(gray, work, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        work = gray.clone();

    const int W = work.cols, H = work.rows;

    // ---------- 2. 背景均衡：减去大尺度背景，抑制光照不均与阴影 ----------
    int ksize = (int)std::lround(std::max(W, H) * 0.12);
    ksize = std::max(15, ksize | 1);                  // 保证奇数且足够大
    cv::Mat bg;
    cv::GaussianBlur(work, bg, cv::Size(ksize, ksize), 0);

    cv::Mat wf, bf, flat;
    work.convertTo(wf, CV_32F);
    bg.convertTo(bf, CV_32F);
    cv::Mat flatf = wf - bf + 128.0;                  // 背景拉回中灰 128
    flatf.convertTo(flat, CV_8U);

    // 轻度中值滤波，去掉传感器的孤立亮点
    cv::medianBlur(flat, flat, 3);

    // ---------- 3. Otsu 分割，同时准备两种极性 ----------
    cv::Mat otsu;
    cv::threshold(flat, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat darkMask  = 255 - otsu;   // 暗物体（深色回形针 / 亮背景）
    cv::Mat lightMask = otsu;         // 亮物体（白色回形针 / 暗背景）

    // ---------- 4. 自动选择极性 ----------
    const double minCompArea = std::max(20.0, 0.00003 * (double)W * (double)H);
    MaskEval evDark  = evaluateMask(darkMask,  minCompArea);
    MaskEval evLight = evaluateMask(lightMask, minCompArea);

    cv::Mat mask;
    if (evDark.valid && !evLight.valid)        { mask = darkMask; }
    else if (evLight.valid && !evDark.valid)   { mask = lightMask; }
    else if (evDark.valid && evLight.valid)    // 两者都说得通时，取"更像细丝环"的那个
    {
        mask = (evDark.solidity <= evLight.solidity) ? darkMask : lightMask;
    }
    else                                       { mask = darkMask; }

    // ---------- 5. 形态学：先闭运算接断点，再开运算去毛刺 ----------
    int closeSize = std::max(3, (int)std::lround(std::max(W, H) * 0.004) | 1);
    cv::Mat kClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closeSize, closeSize));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kClose);

    cv::Mat kOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kOpen);

    // ---------- 6. 连通域统计 ----------
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    if (n <= 1)
    {
        std::cout << 0 << std::endl;
        return 0;
    }

    std::vector<double> rawAreas;
    for (int i = 1; i < n; ++i)
        rawAreas.push_back((double)stats.at<int>(i, cv::CC_STAT_AREA));

    // 噪声地板：与图像尺度挂钩（占全图面积的比例），先剔除碎点再统计，
    // 否则插值振铃/压缩噪点会把"单只基准面积"拉低，造成爆炸式重复计数
    const double minAbsArea = std::max(12.0, 0.00002 * (double)W * (double)H);
    std::vector<double> cleanAreas;
    for (double a : rawAreas)
        if (a >= minAbsArea) cleanAreas.push_back(a);

    const double medArea = cleanAreas.empty() ? medianOf(rawAreas) : medianOf(cleanAreas);

    // 单只基准面积分两步估计：
    //   ① 只保留"与最大块同量级"的连通域（>=15% 最大面积），排除碎点
    //   ② 再剔除大于 1.6 倍中位数的块（粘连体），重算中位数
    double maxArea = 0.0;
    for (double a : cleanAreas) maxArea = std::max(maxArea, a);

    std::vector<double> sameScale;
    for (double a : cleanAreas)
        if (a >= 0.15 * maxArea) sameScale.push_back(a);

    const double roughUnit = sameScale.empty() ? medArea : medianOf(sameScale);
    std::vector<double> singles;
    for (double a : sameScale)
        if (a <= 1.60 * roughUnit) singles.push_back(a);
    const double unit = singles.empty() ? roughUnit : medianOf(singles);

    const double minNoise = std::max(minAbsArea, 0.20 * unit);  // 噪点上限
    const double minFull  = 0.55 * unit;   // 完整回形针下限
    const double minBorder = 0.80 * unit;  // 贴边情况下的完整下限

    // ---------- 7. 计数与标注 ----------
    int count = 0, partialCnt = 0, noiseCnt = 0;
    cv::Mat result = src.clone();

    for (int i = 1; i < n; ++i)
    {
        const double area = (double)stats.at<int>(i, cv::CC_STAT_AREA);
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        if (area < minNoise) { ++noiseCnt; continue; }

        // 是否贴边（贴边又不完整 → 只露出一部分的回形针）
        const bool touchBorder = (x <= 1 || y <= 1 || x + w >= W - 1 || y + h >= H - 1);

        if (area < minFull || (touchBorder && area < minBorder))
        {
            ++partialCnt;
            const cv::Point2f c((float)(centroids.at<double>(i, 0) / scale),
                                (float)(centroids.at<double>(i, 1) / scale));
            cv::putText(result, "partial", cv::Point((int)c.x - 28, (int)c.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(150, 150, 150), 2);
            continue;
        }

        // 面积/基准面积 → 这一块里有几只（粘连会自动得到 2、3…）
        const int numHere = std::max(1, (int)std::lround(area / unit));
        count += numHere;

        // 轮廓搬回原图分辨率再绘制
        std::vector<std::vector<cv::Point>> cs;
        cv::findContours(labels == i, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (auto& c : cs)
            for (auto& p : c)
                p = cv::Point(cvRound(p.x / scale), cvRound(p.y / scale));
        cv::drawContours(result, cs, -1, cv::Scalar(0, 0, 255), 2);

        const cv::Point2f cc((float)(centroids.at<double>(i, 0) / scale),
                             (float)(centroids.at<double>(i, 1) / scale));
        const std::string tag = (numHere > 1) ? ("x" + std::to_string(numHere))
                                              : std::to_string(count);
        // 标签做边界钳制，避免贴边被裁掉
        cv::Point org((int)cc.x - 10, (int)cc.y + 8);
        org.x = std::max(4, std::min(org.x, result.cols - 45));
        org.y = std::max(22, std::min(org.y, result.rows - 6));
        cv::putText(result, tag, org, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 0, 255), 2);
    }

    // ---------- 8. 输出 ----------
    cv::imwrite(savePath, result);
    std::cout << count << std::endl;

    cv::imshow("result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
