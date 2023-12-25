#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class VideoEditingPlan : public Plan {
	cv::UMat frame_;
	const string hv_ = "Hello Video!";
	//Property extends Edge which means it can be directly passed without Edge-directive
	Property<cv::Rect> vp_ = GET<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void infer() override {
		//Capture video from the source
		capture();

		//Render on top of the video
		nvg([](const Rect& vp, const string& str) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(vp.width / 2.0, vp.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, vp_, R(hv_));

		//Write video to the sink
		write();
	}
};

int main(int argc, char** argv) {
	if (argc != 3) {
        std::cerr << "Usage: video_editing <input-video-file> <output-video-file>" << std::endl;
        exit(1);
    }
    cv::Rect viewport(0, 0, 960, 960);
    Ptr<V4D> runtime = V4D::init(viewport, "Video Editing", AllocateFlags::NANOVG | AllocateFlags::IMGUI);

    //Make the video source
    auto src = Source::make(runtime, argv[1]);

    //Make the video sink
    auto sink = Sink::make(runtime, argv[2], src->fps(), viewport.size());

    //Attach source and sink
    runtime->setSource(src);
    runtime->setSink(sink);

    Plan::run<VideoEditingPlan>(0);
}
