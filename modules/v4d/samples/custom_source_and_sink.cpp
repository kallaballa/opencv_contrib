#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class CustomSourceAndSinkPlan : public Plan {
	const string hr_ = "Hello Rainbow!";
	Property<cv::Rect> vp_ = GET<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void infer() override {
		capture();

		//Render "Hello Rainbow!" over the video
		nvg([](const Rect& vp, const string& str) {
			using namespace cv::v4d::nvg;

			fontSize(40.0f);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(vp.width / 2.0, vp.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, vp_, R(hr_));

		write();
	}
};

int main() {
	cv::Rect viewport(0, 0, 960, 960);
    Ptr<V4D> runtime = V4D::init(viewport, "Custom Source/Sink", AllocateFlags::NANOVG | AllocateFlags::IMGUI);

	//Make a source that generates rainbow frames.
	cv::Ptr<Source> src = new Source([](cv::UMat& frame){
	    //The source is responsible for initializing the frame..
		if(frame.empty()) {
			//you may pass a BGR or a BGRA image because BGR is automatically converted to BGRA.
		    frame.create(Size(960, 960), CV_8UC4);
		}
	    frame = colorConvert(Scalar(int64_t(60 * cv::getTickCount() / cv::getTickFrequency()) % 180, 128, 128, 255), COLOR_HLS2BGR);
	    return true;
	}, 60.0f);

	//Make a sink that prints blue, red or green when the current frame is closest to those colors.
	cv::Ptr<Sink> sink = new Sink([](const uint64_t& seq, const cv::UMat& frame){
		static cv::Vec4b last;

		//GPU-copy the top-left pixel to a 1x1 UMat
		cv::UMat onePix;
		frame(cv::Rect(0, 0, 1, 1)).copyTo(onePix);

		//Retrieve the pixel
		const Vec4b topLeft = onePix
				.getMat(cv::ACCESS_READ) // download it
				.at<cv::Vec4b>(0,0); // access it as Vec4b

		//Print only if the top-left pixel has changed and is "pastel" blue, green or red.
		if(topLeft != last) {
			if(topLeft == Vec4b(192, 64, 64, 255)) {
				std::cerr << "Blue" << std::endl;
			} else if(topLeft == Vec4b(64, 192, 64, 255)) {
				std::cerr << "Green" << std::endl;
			} else if(topLeft == Vec4b(64, 64, 192, 255)) {
				std::cerr << "Red" << std::endl;
			}
			last = topLeft;
		}

        return true;
	});

	//Attach source and sink
	runtime->setSource(src);
	runtime->setSink(sink);

	Plan::run<CustomSourceAndSinkPlan>(0);
}

