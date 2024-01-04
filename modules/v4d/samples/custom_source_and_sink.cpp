#include <opencv2/v4d/v4d.hpp>
#include <string>

using namespace cv;
using namespace cv::v4d;

static double seconds() {
	return (cv::getTickCount() / cv::getTickFrequency());
}

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
		//the HLS conversion produces ones where there should be zero values when it produces
		//pure RGB/BGR colors (e.g. 255, 1, 1). Hence the product must be 255 or 255^2
		double prod = pix[0] * pix[1] * pix[2];
		bool isPureColor = prod == 0xff | prod == 0xffff;

		if(lastColor_ != pix && isPureColor) {
			cv::Vec4b binarized = convert_pix<-1, Scalar, cv::Vec4b, true>(pix, 1.0/255);
			uchar key = binarized[0] << 2 | binarized[1] << 1 | binarized[2];
			foundName_ = binarizedBGRIndex_[key - 1];
			found_ = true;
			lastColor_ = pix;
		} else {
			found_ = true;
		}
	}

	void draw(const Rect& vp) const {
		using namespace cv::v4d::nvg;
		string str = "Last pure color: " + foundName_;
		if(!found()) {
			str = "Detecting...";
		}

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

		fb(&PureColor::find, RW(finder_));
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
    cv::Ptr<Sink> videoSink = Sink::make(runtime, "custom_source_and_sink.mkv", 1.0/3.0, viewport.size());

	//Make a source that generates a rainbow frames series.
	cv::Ptr<Source> src = new Source([](cv::UMat& frame){
		//NOTE: The source frame may be generated as RGB or RGBA.

		//The source is responsible for initializing the frame.
		if(frame.empty()) {
		    frame.create(Size(960, 960), CV_8UC3);
		}
		uchar hue = (int64_t(seconds() * 10) % 180);
		//convert from HLS to RGB and set the whole frame to the RGB color
		frame = convert_pix<cv::COLOR_HLS2RGB>(Vec3b(hue, 128, 255));
	    return true; //false signals end of stream (fatal errors should be propagated through exception)
	}, 60.f);

	//Make a sink that prints the main color of the frame and passes the frame to the video sink
	cv::Ptr<Sink> sink = new Sink([videoSink](const uint64_t& seq, const cv::UMat& frame){
		//NOTE. In sinks the frame is always RGBA

		//we could do all kinds of operations and decisions herem that are based
		//on the frame, the sequence nummber and any  hidden state the sink holds
		//(e.g. the video sink)
		Scalar pix = cv::sum(frame(cv::Rect(0, 0, 1, 1)));
		//simply print the (pure) main frame color.
		std::cerr << "WRITE: " << pix << std::endl;
		//pass the frame on to the video sink
		videoSink->operator()(seq, frame);
		return  videoSink->isOpen(); // false signals a temporary error. (fatal errors should be propagated through exceptions).
	});

	//Attach source and sink
	runtime->setSource(src);
	runtime->setSink(sink);

	Plan::run<CustomSourceAndSinkPlan>(0);
}

