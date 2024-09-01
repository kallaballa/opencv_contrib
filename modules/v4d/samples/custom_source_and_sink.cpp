#include <opencv2/v4d/v4d.hpp>
#include <string>

using namespace cv;
using namespace cv::v4d;

class PureColor {
	Scalar lastColor_;
	string foundName_;
	bool found_;
	std::vector<std::string> binarizedBGRIndex_ = {
			"RED",    //0,  0,255 -> 0,0,1 -> 0b001 - 1 == 0
			"GREEN",  //0,255,  0 -> 0,1,0 -> 0b010 - 1 == 1
			"YELLOW", //0,255,255 -> 0,1,1 -> 0b011 - 1 == 2
			"BLUE",   //...
			"FUCHSIA",
			"AQUA"
	};
public:
	void find(const cv::UMat& bgra) {
		//NOTE: framebuffer is always BGRA no matter what format the source generates.

		//Retrieve the bottom-left pixel (using getMat on a aub-UMat downloads the whole frame)
		Scalar pix = cv::sum(bgra(cv::Rect(0, 0, 1, 1)));

		double sum = pix[0] + pix[1] + pix[2];
		bool isPureColor = sum == 257 || sum == 508;


		if(lastColor_ != pix && isPureColor) {
			std::cerr << pix << std::endl;
			cv::Vec4b binarized = convert_pix<-1, Scalar, cv::Vec4b, true>(pix, 1.0/255.0);
			std::cerr << binarized << std::endl;
			uchar key = binarized[0] | binarized[1] << 1 | binarized[2] << 2;
			std::cerr << key - 1 << std::endl;
			foundName_ = binarizedBGRIndex_[key - 1];
			found_ = true;
			lastColor_ = pix;
		} else {
			found_ = false;
		}
	}

	void draw(const Rect& vp) const {
		using namespace cv::v4d::nvg;
		string str = "Last detected color: " + foundName_;

		fontSize(40.0f);
		fontFace("sans-bold");
		fillColor(Scalar(127, 127, 127, 255));
		textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
		text(vp.width / 2.0, vp.height / 2.0, str.c_str(), str.c_str() + str.size());
	}

	bool found() const {
		return found_;
	}
};

class CustomSourceAndSinkPlan : public Plan {
	struct Params {
		cv::Scalar last_;
		std::string colorName_;
		bool found_;
	} params_;

	PureColor finder_;
	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void infer() override {
		capture();

		fb<1>(&PureColor::find, RW(finder_));
		nvg(&PureColor::draw, R(finder_), vp_);

		branch(&PureColor::found, R(finder_))
			->write()
		->endBranch();
	}
};


int main() {
	cv::Rect viewport(0, 0, 960, 960);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Custom Source/Sink", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
    //check out the video after. It will only contain frames filtered by the plan by conditional branching
    //the frame rate is set to one frame every 3 seconds because that is what we are going to emit to the video.
    //anyway, xou may choose a fps value to your own liking.
    cv::Ptr<Sink> videoSink = Sink::make(runtime, "custom_source_and_sink.mkv", 10, viewport.size());

	//Make a source that generates a rainbow frames series.
	cv::Ptr<Source> src = new Source([](cv::UMat& frame){
		//NOTE: The source frame may be generated as RGB or RGBA.

		//The source is responsible for initializing the frame.
		if(frame.empty()) {
		    frame.create(Size(960, 960), CV_8UC3);
		}
		uchar hue = (int64_t(seconds() * 15) % 255);

		//convert from HLS to RGB and set the whole frame to the RGB color
		frame = convert_pix<cv::COLOR_HLS2RGB_FULL>(cv::Vec3b(hue, 128, 255));
	    return true; //false signals end of stream (fatal errors should be propagated through exception)
	}, 60.f);

	//Make a sink that prints the main color of the frame and passes the frame to the video sink
	cv::Ptr<Sink> sink = new Sink([videoSink](const uint64_t& seq, const cv::UMat& frame){
		//NOTE. In sinks the frame is always RGBA

		//we could do all kinds of operations and decisions herem that are based
		//on the frame, the sequence nummber and any  hidden state the sink holds
		//(e.g. the video sink)
		Scalar pix = cv::sum(frame(cv::Rect(0, 0, 1, 1)));

		//pass the frame on to the video sink
		videoSink->operator()(seq, frame);
		return  videoSink->isOpen(); // false signals a temporary error. (fatal errors should be propagated through exceptions).
	});

	//Attach source and sink
	runtime->setSource(src);
	runtime->setSink(sink);

	Plan::run<CustomSourceAndSinkPlan>(0);
}

