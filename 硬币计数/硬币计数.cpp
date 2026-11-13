#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char* argv[])
{
    // 图片路径：默认当前工作目录下的 2.png，也可通过命令行参数传入
    std::string imagePath = (argc > 1) ? argv[1] : "2.png";
    std::string savePath = "coin_result.png";

    cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cerr << "无法读取图片: " << imagePath << std::endl;
        return -1;
    }

    // ---------- 1. 预处理 ----------
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);

    // ---------- 2. 霍夫圆检测 ----------
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
                     1,            // 分辨率比
                     60,           // 圆心最小间距（像素），防止重复检测
                     120,          // Canny 高阈值
                     35,           // 累加器阈值：越小越容易检出（也可能误检）
                     30,           // 最小半径
                     85);          // 最大半径

    // ---------- 3. 去重：圆心距过近的候选只保留一个 ----------
    std::vector<cv::Vec3f> kept;
    std::sort(circles.begin(), circles.end(),
              [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] > b[2]; });
    for (const auto& c : circles)
    {
        bool dup = false;
        for (const auto& k : kept)
        {
            double d = cv::norm(cv::Point2f(c[0], c[1]) - cv::Point2f(k[0], k[1]));
            if (d < 0.5 * std::min(c[2], k[2])) { dup = true; break; }
        }
        if (!dup) kept.push_back(c);
    }

    // ---------- 4. 画结果 ----------
    cv::Mat result = src.clone();
    int count = 0;
    for (const auto& c : kept)
    {
        count++;
        cv::Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        cv::circle(result, center, radius, cv::Scalar(0, 0, 255), 2);
        cv::circle(result, center, 3, cv::Scalar(0, 255, 0), -1);

        // 编号位置：默认放在圆的左上方，但钳制在图片范围内，防止被边缘截断
        int lx = cvRound(c[0] - c[2]);
        int ly = cvRound(c[1] - c[2] - 4);
        lx = std::max(2, std::min(lx, src.cols - 45));
        ly = std::max(22, std::min(ly, src.rows - 12));
        cv::putText(result, std::to_string(count),
                    cv::Point(lx, ly),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    // ---------- 5. 输出 ----------
    cv::imwrite(savePath, result);
    std::cout << "检测到的硬币数量: " << count << std::endl;
    std::cout << "标注结果已保存到: " << savePath << std::endl;

    cv::imshow("result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
