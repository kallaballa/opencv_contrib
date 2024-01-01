#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class FontRenderingPlan: public Plan {
	//The text to render
	string text_ = "Hello World";
	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void infer() override {
		//Render the text at the center of the screen. Note that you can load you own fonts.
		nvg([](const Rect& vp, const string& str) {
			using namespace cv::v4d::nvg;
			clearScreen();
			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(vp.width / 2.0, vp.height / 2.0, str.c_str(),
					str.c_str() + str.size());
		}, vp_, R(text_));
	}
};

int main() {
	cv::Rect viewport(0, 0, 960, 960);
	cv::Ptr<V4D> runtime = V4D::init(viewport, "Font Rendering", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
	Plan::run<FontRenderingPlan>(0);

	return 0;
}
