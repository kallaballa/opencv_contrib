#include <opencv2/v4d/v4d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::v4d;

class DisplayImageNVG : public Plan {
	using K = V4D::Keys;
	//A simple struct to hold our image variables
	struct Image_t {
	    std::string filename_;
	    nvg::Paint paint_;
	    int w_;
	    int h_;
	} image_;
public:
	DisplayImageNVG(const string& filename) {
		//Set the filename
		image_.filename_ = filename;
	}

	void setup() override {
		//Creates a NanoVG context. The wrapped C-functions of NanoVG are available in the namespace cv::v4d::nvg;
		nvg([](Image_t& img) {
			using namespace cv::v4d::nvg;
			//Create the image_ and receive a handle.
			int handle = createImage(img.filename_.c_str(), NVG_IMAGE_NEAREST);
			//Make sure it was created successfully
			CV_Assert(handle > 0);
			//Query the image_ size
			imageSize(handle, &img.w_, &img.h_);
			//Create a simple image_ pattern with the image dimensions
			img.paint_ = imagePattern(0, 0, img.w_, img.h_, 0.0f/180.0f*NVG_PI, handle, 1.0);
		}, A(image_));
	}

	void infer() override{
		//Creates a NanoVG context to draw the loaded image_ over again to the screen.
		nvg([](const cv::Rect& vp, const Image_t& img) {
			using namespace cv::v4d::nvg;
			beginPath();
			//Scale all further calls to window size
			scale(double(vp.width)/img.w_, double(vp.height)/img.h_);
			//Create a rounded rectangle with the images dimensions.
			//Note that actually this rectangle will have the size of the window
			//because of the previous scale call.
			roundedRect(0,0, img.w_, img.h_, 50);
			//Fill the rounded rectangle with our picture
			fillPaint(img.paint_);
			fill();
		}, P<cv::Rect>(K::VIEWPORT), A(image_));
	}
};

int main() {
	cv::Rect viewport(0, 0, 960, 960);
	Ptr<V4D> runtime = V4D::init(viewport, "Display an image using NanoVG", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
    Plan::run<DisplayImageNVG>(0, samples::findFile("lena.jpg"));
}
