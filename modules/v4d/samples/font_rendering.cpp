#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class FontRenderingPlan: public Plan {
	//The text to render
	string hw_ = "Hello World";
public:
	using Plan::Plan;

	void infer(Ptr<V4D> win) override {
		//Render the text at the center of the screen. Note that you can load you own fonts.
		win->nvg([](const Size &sz, const string &str) {
			using namespace cv::v4d::nvg;
			clear();
			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, str.c_str(),
					str.c_str() + str.size());
		}, R(size()), R(hw_));
	}
};

int main() {
	cv::Rect viewport(0, 0, 960, 960);
	cv::Ptr<V4D> window = V4D::make(viewport.size(), "Font Rendering", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
	window->run<FontRenderingPlan>(0, viewport);
}
