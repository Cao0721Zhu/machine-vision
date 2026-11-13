#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

static double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv)
{
    std::string path = (argc > 1) ? argv[1] : "3.png";
    cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) { std::cerr << "无法读取图片: " << path << std::endl; return -1; }

    // 1. 尺度归一：长边缩到 900px 以内，参数与分辨率解耦
    double scale = 1.0;
    cv::Mat gray = src;
    if (std::max(src.cols, src.rows) > 900)
    {
        scale = 900.0 / (double)std::max(src.cols, src.rows);
        cv::resize(src, gray, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    const int W = gray.cols, H = gray.rows;
    const int longSide = std::max(W, H);

    // 2. 背景均衡：减去大尺度背景（核取长边 30%），再轻度模糊抹掉表面高光
    int kBg = std::max(15, (int)std::lround(longSide * 0.30) | 1);
    int kBlur = std::max(3, (int)std::lround(longSide * 0.012) | 1);
    cv::Mat bg, gf, bf, flat;
    cv::GaussianBlur(gray, bg, cv::Size(kBg, kBg), 0);
    gray.convertTo(gf, CV_32F);
    bg.convertTo(bf, CV_32F);
    cv::Mat flatf = gf - bf + 128.0;
    flatf.convertTo(flat, CV_8U);
    cv::GaussianBlur(flat, flat, cv::Size(kBlur, kBlur), 0);

    // 3. Otsu 分割。物体总是"少数派"：前景占比小的那个极性才是孔/硬币
    cv::Mat otsu;
    cv::threshold(flat, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat dark = 255 - otsu, light = otsu;
    cv::Mat mask = (cv::countNonZero(dark) <= cv::countNonZero(light)) ? dark : light;

    // 4. 闭运算补小缺口，开运算去毛刺
    int kClose = std::max(3, (int)std::lround(longSide * 0.004) | 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kClose, kClose)));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    // 5. 连通域 + 逐块几何量（圆形度与尺寸无关，可先于基准面积计算）
    cv::Mat labels, stats, cent;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, cent, 8, CV_32S);
    const double minAbsArea = std::max(12.0, 0.00002 * (double)W * (double)H);

    struct Blob { double area; bool edge; double circ;
                  std::vector<std::vector<cv::Point>> cs;
                  cv::Point2f c; float r; };
    std::vector<Blob> blobs;
    for (int i = 1; i < n; ++i)
    {
        Blob b;
        b.area = (double)stats.at<int>(i, cv::CC_STAT_AREA);
        if (b.area < minAbsArea) continue;
        cv::findContours(labels == i, b.cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (b.cs.empty()) continue;
        const double peri = cv::arcLength(b.cs[0], true);
        b.circ = (peri > 0) ? 4.0 * CV_PI * b.area / (peri * peri) : 0.0;
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        b.edge = (x <= 1 || y <= 1 || x + w >= W - 1 || y + h >= H - 1);
        cv::minEnclosingCircle(b.cs[0], b.c, b.r);
        blobs.push_back(b);
    }

    // 6. 基准面积：只在"够圆且完整在框内"的候选里统计（边框、不规则大块
    //    已被圆形度排除，不会污染最大值）。两阶段中位数：先排除小斑点，
    //    再排除约 2 倍的粘连块
    std::vector<double> ca;
    double candMax = 0.0;
    for (const auto& b : blobs)
        if (!b.edge && b.circ >= 0.75)
        {
            ca.push_back(b.area);
            candMax = std::max(candMax, b.area);
        }
    std::vector<double> core;
    for (double a : ca)
        if (a >= 0.15 * candMax) core.push_back(a);      // 先排除小斑点
    const double rough = median(core);
    std::vector<double> singles;
    for (double a : core)
        if (a <= 1.60 * rough) singles.push_back(a);     // 再排除约 2 倍的粘连块
    const double unit = singles.empty() ? rough : median(singles);   // 单个圆孔的基准面积
    if (unit <= 0.0) { std::cout << 0 << std::endl; return 0; }
    const double lo = 0.25 * unit, hi = 2.00 * unit;

    // 7. 分类绘制：主簇成员=完整圆孔；贴边且大致圆的标 partial
    std::vector<cv::Vec3f> found;
    cv::Mat result;
    cv::cvtColor(src, result, cv::COLOR_GRAY2BGR);
    for (const auto& b : blobs)
    {
        if (b.edge)
        {
            if (b.circ >= 0.5 && b.area >= 0.15 * unit)
            {
                std::vector<std::vector<cv::Point>> cs = b.cs;
                for (auto& c : cs) for (auto& p : c)
                    p = cv::Point(cvRound(p.x / scale), cvRound(p.y / scale));
                cv::drawContours(result, cs, -1, cv::Scalar(160, 160, 160), 1);
                cv::putText(result, "partial",
                            cv::Point(cvRound(b.c.x / scale) - 28, cvRound(b.c.y / scale)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(160, 160, 160), 2);
            }
            continue;
        }
        if (b.circ >= 0.75 && b.area >= lo && b.area <= hi)
            found.push_back(cv::Vec3f(b.c.x / scale, b.c.y / scale, b.r / scale));
    }

    // 8. 按从上到下排序编号并绘制
    std::sort(found.begin(), found.end(),
              [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[1] < b[1]; });
    int count = 0;
    for (const auto& f : found)
    {
        cv::Point center(cvRound(f[0]), cvRound(f[1]));
        int r = cvRound(f[2]) + 4;
        ++count;
        cv::circle(result, center, cvRound(f[2]), cv::Scalar(255, 100, 0), 2);
        cv::circle(result, center, 3, cv::Scalar(0, 0, 255), -1);
        cv::Point org = center + cv::Point(-r, -r - 4);
        org.x = std::max(2, std::min(org.x, result.cols - 40));
        org.y = std::max(18, std::min(org.y, result.rows - 6));
        cv::putText(result, std::to_string(count), org,
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    // 9. 输出
    std::cout << count << std::endl;
    cv::imwrite("hole_result.png", result);

    cv::imshow("result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
