#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

using namespace cv;
using namespace cv::v4d;

std::map<uchar, std::string> base_colors = {
		{1, "RED"},
		{2, "GREEN"},
		{3, "YELLOW"},
		{4, "BLUE"},
		{5, "FUCHSIA"},
		{6, "AQUA"},
};

static double seconds() {
	return (cv::getTickCount() / cv::getTickFrequency());
}

class CustomSourceAndSinkPlan : public Plan {
	struct Params {
		cv::Scalar last_;
		std::string colorName_;
		bool writeFrame_;
	} params_;

	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void infer() override {
		capture();

		fb([](const cv::UMat& framebuffer, Scalar& last, string& colorName, bool& writeFrame) {
			//Retrieve the bottom-left pixel (using getMat on a aub-UMat downloads the whole frame)
			Scalar pix = cv::sum(framebuffer(cv::Rect(0, 0, 1, 1)));
			double prod = pix[0] * pix[1] * pix[2];
			bool isPureColor = sum(pix)[0] > 0 && fmod(prod - 255.0, pow(255.0, 2.0) - 255.0) == 0;

			if(last != pix && isPureColor) {
				cv::Vec4b binarized = convert_pix<-1, Scalar, cv::Vec4b, true>(pix, 1.0/255);
				uchar key = binarized[2] << 2 | binarized[1] << 1 | binarized[0];
				colorName = base_colors[key];
				writeFrame = true;
				last = pix;
			} else {
				writeFrame = false;
			}
		}, RW(params_.last_), RW(params_.colorName_), RW(params_.writeFrame_));

		//Render "Hello Rainbow!" over the video
		nvg([](const Rect& vp, const string& colorName) {
			using namespace cv::v4d::nvg;
			const string str = "Last Pure Color: " + colorName;
			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(0, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(vp.width / 2.0, vp.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, vp_, R(params_.colorName_));

		branch(isTrue_, R(params_.writeFrame_))
			->write()
		->endBranch();
	}
};


int main() {
	cv::Rect viewport(0, 0, 960, 960);
    Ptr<V4D> runtime = V4D::init(viewport, "Custom Source/Sink", AllocateFlags::NANOVG | AllocateFlags::IMGUI);

	//Make a source that generates a rainbow frames series.
	cv::Ptr<Source> src = new Source([](cv::UMat& frame){
	    //The source is responsible for initializing the frame.
		if(frame.empty()) {
			//The source expects a RGB or a RGBA image.
		    frame.create(Size(960, 960), CV_8UC3);
		}
		uchar hue = (int64_t(seconds() * 10) % 180);
		frame = convert_pix<cv::COLOR_HLS2RGB>(Vec3b(hue, 128, 255));
	    return true; //false signals end of stream (fatal errors should be propagated through exception)
	}, 60.f);

	//Make a sink that prints rainbow colors when they are most saturated.
	cv::Ptr<Sink> sink = new Sink([](const uint64_t& seq, const cv::UMat& frame){
		cv::Scalar m = cv::mean(frame) / 255.0;
		std::cerr << "WRITE: ("
				<< std::round(m[0]) * 255.0 << ", "
				<< std::round(m[1]) * 255.0 << ", "
				<< std::round(m[2]) * 255.0 << ")" << std::endl;
		return true; // false signals a temporary error. (fatal errors should be propagated through exceptions).
	});

	//Attach source and sink
	runtime->setSource(src);
	runtime->setSink(sink);

	Plan::run<CustomSourceAndSinkPlan>(0);
}

