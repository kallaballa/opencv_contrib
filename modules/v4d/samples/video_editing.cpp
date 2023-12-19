#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class VideoEditingPlan : public Plan {
	cv::UMat frame_;
	const string hv_ = "Hello Video!";
public:
	using Plan::Plan;
	void infer(Ptr<V4D> win) override {
		//Capture video from the source
		win->capture();

		//Render on top of the video
		win->nvg([](const Size& sz, const string& str) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, R(size()), R(hv_));

		//Write video to the sink
		win->write();
	}
};

int main(int argc, char** argv) {
	if (argc != 3) {
        cerr << "Usage: video_editing <input-video-file> <output-video-file>" << endl;
        exit(1);
    }
    cv::Rect viewport(0, 0, 960, 960);
    Ptr<V4D> window = V4D::make(viewport.size(), "Video Editing", AllocateFlags::NANOVG | AllocateFlags::IMGUI);

    //Make the video source
    auto src = Source::make(window, argv[1]);

    //Make the video sink
    auto sink = Sink::make(window, argv[2], src->fps(), viewport.size());

    //Attach source and sink
    window->setSource(src);
    window->setSink(sink);

    window->run<VideoEditingPlan>(0, viewport);
}

