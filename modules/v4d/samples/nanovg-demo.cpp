// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

static void draw_color_wheel(float x, float y, float w, float h, double hue) {
    //color wheel drawing code taken from https://github.com/memononen/nanovg/blob/master/example/demo.c
    using namespace cv::v4d::nvg;
    int i;
    float r0, r1, ax, ay, bx, by, cx, cy, aeps, r;
    Paint paint;

    save();

    cx = x + w * 0.5f;
    cy = y + h * 0.5f;
    r1 = (w < h ? w : h) * 0.5f - 5.0f;
    r0 = r1 - 20.0f;
    aeps = 0.5f / r1;   // half a pixel arc length in radians (2pi cancels out).

    for (i = 0; i < 6; i++) {
        float a0 = (float) i / 6.0f * CV_PI * 2.0f - aeps;
        float a1 = (float) (i + 1.0f) / 6.0f * CV_PI * 2.0f + aeps;
        beginPath();
        arc(cx, cy, r0, a0, a1, NVG_CW);
        arc(cx, cy, r1, a1, a0, NVG_CCW);
        closePath();
        ax = cx + cosf(a0) * (r0 + r1) * 0.5f;
        ay = cy + sinf(a0) * (r0 + r1) * 0.5f;
        bx = cx + cosf(a1) * (r0 + r1) * 0.5f;
        by = cy + sinf(a1) * (r0 + r1) * 0.5f;
        paint = linearGradient(ax, ay, bx, by,
                cv::v4d::convert_pix<cv::COLOR_HLS2BGR, cv::Vec3b, cv::Vec4f>(cv::Vec3b((a0 / (CV_PI * 2.0)) * 180, 0.55 * 255, 255)),
                cv::v4d::convert_pix<cv::COLOR_HLS2BGR, cv::Vec3b, cv::Vec4f>(cv::Vec3b((a1 / (CV_PI * 2.0)) * 180, 0.55 * 255, 255)));
        fillPaint(paint);
        fill();
    }

    beginPath();
    circle(cx, cy, r0 - 0.5f);
    circle(cx, cy, r1 + 0.5f);
    strokeColor(cv::Scalar(0, 0, 0, 64));
    strokeWidth(1.0f);
    stroke();

    // Selector
    save();
    translate(cx, cy);
    rotate((hue/255.0) * CV_PI * 2);

    // Marker on
    strokeWidth(2.0f);
    beginPath();
    rect(r0 - 1, -3, r1 - r0 + 2, 6);
    strokeColor(cv::Scalar(255, 255, 255, 192));
    stroke();

    paint = boxGradient(r0 - 3, -5, r1 - r0 + 6, 10, 2, 4, cv::Scalar(0, 0, 0, 128), cv::Scalar(0, 0, 0, 0));
    beginPath();
    rect(r0 - 2 - 10, -4 - 10, r1 - r0 + 4 + 20, 8 + 20);
    rect(r0 - 2, -4, r1 - r0 + 4, 8);
    pathWinding(NVG_HOLE);
    fillPaint(paint);
    fill();

    // Center triangle
    r = r0 - 6;
    ax = cosf(120.0f / 180.0f * NVG_PI) * r;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r;
    bx = cosf(-120.0f / 180.0f * NVG_PI) * r;
    by = sinf(-120.0f / 180.0f * NVG_PI) * r;
    beginPath();
    moveTo(r, 0);
    lineTo(ax, ay);
    lineTo(bx, by);
    closePath();
    paint = linearGradient(r, 0, ax, ay, cv::v4d::convert_pix<cv::COLOR_HLS2BGR_FULL, cv::Vec3b, cv::Vec4f>(cv::Vec3b(uchar(hue), 128, 255)), cv::Scalar(255, 255, 255, 255));
    fillPaint(paint);
    fill();
    paint = linearGradient((r + ax) * 0.5f, (0 + ay) * 0.5f, bx, by, cv::Scalar(0, 0, 0, 0), cv::Scalar(0, 0, 0, 255));
    fillPaint(paint);
    fill();
    strokeColor(cv::Scalar(0, 0, 0, 64));
    stroke();

    // Select circle on triangle
    ax = cosf(120.0f / 180.0f * NVG_PI) * r * 0.3f;
    ay = sinf(120.0f / 180.0f * NVG_PI) * r * 0.4f;
    strokeWidth(2.0f);
    beginPath();
    circle(ax, ay, 5);
    strokeColor(cv::Scalar(255, 255, 255, 192));
    stroke();

    paint = radialGradient(ax, ay, 7, 9, cv::Scalar(0, 0, 0, 64), cv::Scalar(0, 0, 0, 0));
    beginPath();
    rect(ax - 20, ay - 20, 40, 40);
    circle(ax, ay, 7);
    pathWinding(NVG_HOLE);
    fillPaint(paint);
    fill();

    restore();

    restore();
}

using namespace cv::v4d;

class NanoVGDemoPlan : public Plan {
	std::vector<cv::UMat> hsvChannels_;
	cv::UMat rgb_;
	cv::UMat bgra_;
	cv::UMat hsv_;
	cv::UMat hueChannel_;
	double hue_ = 0;
	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
	size_t width_ = 0;
	size_t height_ = 0;

	constexpr static auto SPLIT_ = _OL_(void, cv::split, cv::InputArray, cv::OutputArrayOfArrays);
	constexpr static auto MERGE_ = _OL_(void, cv::merge, cv::InputArrayOfArrays, cv::OutputArray);
public:
	NanoVGDemoPlan() {
	}

	void setup() override {
		plain(&std::vector<cv::UMat>::reserve, RW(hsvChannels_), V(3));
	}
	void infer() override {
		capture();

		assign(RW(hue_), R((sinf(cv::getTickCount() / cv::getTickFrequency() * 0.12) + 1.0) * 127.5));

		//Acquire the framebuffer and convert it to RGB
		fb(cv::cvtColor, RW(rgb_), V(cv::COLOR_BGRA2RGB), V(0), V(cv::ALGO_HINT_DEFAULT));

		//Transform HSV space
		plain(cv::cvtColor, R(rgb_), RW(hsv_), V(cv::COLOR_RGB2HSV_FULL), V(0), V(cv::ALGO_HINT_DEFAULT))
		->plain(SPLIT_, R(hsv_), RW(hsvChannels_))
		->plain(&cv::UMat::setTo, RW(hsvChannels_[0]), V(std::round(hue_)), V(cv::noArray()))
		->plain(MERGE_, R(hsvChannels_), RW(hsv_))
		->plain(cv::cvtColor, R(hsv_), RW(rgb_), V(cv::COLOR_HSV2RGB_FULL), V(0), V(cv::ALGO_HINT_DEFAULT));

		//Acquire the framebuffer and convert rgb_ into it
		fb<1>(cv::cvtColor, RW(rgb_), V(cv::COLOR_BGR2BGRA), V(0), V(cv::ALGO_HINT_DEFAULT));

		assign(RW(width_), F(&cv::Rect::width, vp_))
		->assign(RW(height_), F(&cv::Rect::height, vp_));

		//Render using nanovg
		nvg(draw_color_wheel, V(width_ - (width_ / 5)), V(height_ - (width_ / 5)), V(width_ / 6), V(width_ / 6), V(hue_));
	}
};

int main(int argc, char **argv) {
	if (argc != 2) {
        std::cerr << "Usage: nanovg-demo <video-file>" << std::endl;
        exit(1);
	}

    cv::Rect viewport(0, 0, 1280, 720);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "NanoVG Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
    auto src = Source::make(runtime, argv[1]);
    runtime->setSource(src);
    Plan::run<NanoVGDemoPlan>(0);

    return 0;
}
