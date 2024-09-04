// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/optflow.hpp>

#include <cmath>
#include <vector>
#include <set>
#include <string>
#include <random>
#include <tuple>
#include <array>
#include <utility>

using std::vector;
using std::string;

using namespace cv::v4d;

struct GlowEffect {
	struct Temp {
		cv::UMat src_;
		cv::UMat dst_;
		cv::UMat dst16_;
		cv::UMat high_;
		cv::UMat blur_;
		cv::UMat low_;
	} temp_;
public:

	//Glow post-processing effect
	void perform(const cv::UMat& srcFloat, cv::UMat& dstFloat, const int ksize) {
		srcFloat.convertTo(temp_.src_, CV_8U, 127.0);

		cv::bitwise_not(temp_.src_, temp_.dst_);

	    //Resize for some extra performance
	    cv::resize(temp_.dst_, temp_.low_, cv::Size(), 0.5, 0.5);
	    //Cheap blur
	    cv::boxFilter(temp_.low_, temp_.blur_, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
	    //Back to original size
	    cv::resize(temp_.blur_, temp_.high_, srcFloat.size());

	    //Multiply the src with a blurred version of itself and convert back to CV_8U
	    cv::multiply(temp_.dst_, temp_.high_, temp_.dst_, 1.0/255.0, CV_8U);

	    cv::bitwise_not(temp_.dst_, temp_.dst_);

	    temp_.dst_.convertTo(dstFloat, CV_32F, 1.0/255.0);
	}
};

struct BloomEffect {
	struct Temp {
		cv::UMat bgr_;
		cv::UMat hls_;
		cv::UMat ls16_;
		cv::UMat ls_;
		cv::UMat blur_;
		std::vector<cv::UMat> hlsChannels_;
	} temp_;
public:
	//Bloom post-processing effect
	void perform(const cv::UMat& srcFloat, cv::UMat &dstFloat, int ksize = 3, int threshValue = 235, float gain = 4) {
	    //remove alpha channel
	    cv::cvtColor(srcFloat, temp_.bgr_, cv::COLOR_BGRA2RGB);
	    //convert to hls
	    cv::cvtColor(temp_.bgr_, temp_.hls_, cv::COLOR_BGR2HLS);
	    //split channels
	    cv::split(temp_.hls_, temp_.hlsChannels_);
	    //invert lightness
	    cv::bitwise_not(temp_.hlsChannels_[2], temp_.hlsChannels_[2]);
	    //multiply lightness and saturation
	    cv::multiply(temp_.hlsChannels_[1], temp_.hlsChannels_[2], temp_.ls_, 255.0, CV_8U);
	    //binary threhold according to threshValue
	    cv::threshold(temp_.ls_, temp_.blur_, threshValue, 255, cv::THRESH_BINARY);
	    //blur
	    cv::boxFilter(temp_.blur_, temp_.blur_, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
	    //convert to BGRA
	    cv::cvtColor(temp_.blur_, temp_.blur_, cv::COLOR_GRAY2BGRA);
	    //add src and the blurred L-S-product according to gain
	    addWeighted(srcFloat, 1.0, temp_.blur_, gain, 0, dstFloat);
	}
};

struct PostProcessor {
	GlowEffect glow_;
	BloomEffect bloom_;
public:
	//Post-processing modes for the foreground
	enum Modes {
	    GLOW,
	    BLOOM,
	    DISABLED
	};

	void perform(const cv::UMat& srcFloat, cv::UMat& dstFloat, const Modes& mode, const int& ksize, const int& bloomThresh, const int& bloomGain) {
	    switch (mode) {
	    case GLOW:
	        glow_.perform(srcFloat, dstFloat, ksize);
	        break;
	    case BLOOM:
	        bloom_.perform(srcFloat, dstFloat, ksize, bloomThresh, bloomGain);
	        break;
	    case DISABLED:
	        srcFloat.copyTo(dstFloat);
	        break;
	    default:
	        break;
	    }
	}
};

class FeaturePoints {
	cv::Ptr<cv::FastFeatureDetector> detector_;
	vector<cv::KeyPoint> tmpKeyPoints_;
public:
	FeaturePoints() {
	}

	FeaturePoints(cv::Ptr<cv::FastFeatureDetector> detector) : detector_(detector) {
	}

	void detect(const cv::UMat& src, vector<cv::Point2f>& output) {
		detector_->detect(src, tmpKeyPoints_);

	    output.clear();
	    for (const auto &kp : tmpKeyPoints_) {
	        output.push_back(kp.pt);
	    }
//	    std::cerr << "detected: " << output.size() << std::endl;
	}
};

class SceneChange {
	float lastMovement_ = 0;
public:
	bool detect(const cv::UMat& motionMask, const float& sceneChangeThresh, const float& sceneChangeThreshDiff) {
	    float movement = cv::countNonZero(motionMask) / float(motionMask.cols * motionMask.rows);
	    float relation = movement > 0 && lastMovement_ > 0 ? std::max(movement, lastMovement_) / std::min(movement, lastMovement_) : 0;
	    float relM = relation * log10(1.0f + (movement * 9.0));
	    float relLM = relation * log10(1.0f + (lastMovement_ * 9.0));

	    bool result = ((movement > 0 && lastMovement_ > 0 && relation > 0)
	            && (relM < sceneChangeThresh && relLM < sceneChangeThresh && fabs(relM - relLM) < sceneChangeThreshDiff));
	    lastMovement_ = (lastMovement_ + movement) / 2.0f;
	    return !result;
	}
};

class BackgroundStyle {
	struct Temp {
	    cv::UMat tmp_;
	    cv::UMat post_;
	    cv::UMat backgroundGrey_;
	    vector<cv::UMat> channels_;
	} temp_;

public:
	enum Modes {
	    GREY,
	    COLOR,
	    VALUE,
	    BLACK
	};

	void apply(const cv::UMat& srcFloat, cv::UMat& dstFloat, const Modes& bgMode) {
	    //Dependin on bgMode prepare the background in different ways

		switch (bgMode) {
	    case GREY:
	        cv::cvtColor(srcFloat, temp_.backgroundGrey_, cv::COLOR_BGRA2GRAY);
	        cv::cvtColor(temp_.backgroundGrey_, dstFloat, cv::COLOR_GRAY2BGRA);
	        break;
	    case VALUE:
	        cv::cvtColor(srcFloat, temp_.tmp_, cv::COLOR_BGRA2BGR);
	        cv::cvtColor(temp_.tmp_, temp_.tmp_, cv::COLOR_BGR2HSV);
 	        split(temp_.tmp_, temp_.channels_);
	        cv::cvtColor(temp_.channels_[2], dstFloat, cv::COLOR_GRAY2BGRA);
	        break;
	    case COLOR:
	    	srcFloat.copyTo(dstFloat);
	        break;
	    case BLACK:
	    	dstFloat = cv::Scalar::all(0);
	        break;
	    default:
	        break;
	    }
	}
};

class Compositor {
	BackgroundStyle backgroundStyle_;
	PostProcessor postProcessor_;

	struct Temp {
		cv::UMat onesFloat_;
		cv::UMat bgFloat_;
		cv::UMat fgFloat_;
		cv::UMat fgInvertedFloat_;
		cv::UMat fbFloat_;
	} temp_;
public:
	//Compose the different layers into the final image
	void perform(const cv::UMat& background, cv::UMat& foreground, cv::UMat& framebuffer, const BackgroundStyle::Modes& bgMode, const PostProcessor::Modes& ppMode, const int& ksize, const int& bloomThresh, const int& bloomGain, const float& fgLoss, const size_t& numWorkers) {
		if(temp_.onesFloat_.empty()) {
			temp_.onesFloat_ = cv::UMat(framebuffer.size(), CV_32FC4, cv::Scalar(1));
		}
		foreground.convertTo(temp_.fgFloat_, CV_32F, 1.0/255.0);
		background.convertTo(temp_.bgFloat_, CV_32F, 1.0/255.0);
		double loss = 1.0 - (fgLoss / 100.0);
		cv::multiply(temp_.fgFloat_, cv::Scalar::all(loss), temp_.fgFloat_);
		backgroundStyle_.apply(temp_.bgFloat_, temp_.bgFloat_, bgMode);
	    postProcessor_.perform(temp_.fgFloat_, temp_.fgFloat_, ppMode, ksize, bloomThresh, bloomGain);
//	    cv::multiply(temp_.fgFloat_, cv::Scalar::all(3), temp_.fgFloat_);
	    cv::add(temp_.bgFloat_, temp_.fgFloat_, temp_.fbFloat_);
	    temp_.fbFloat_.convertTo(framebuffer, CV_8U, 255.0);
	    temp_.fgFloat_.convertTo(foreground, CV_8U, 255.0);
	}
};

class SparseOpticalFlow {
	struct Temp {
		vector<cv::Point2f> hull_, nextPoints_, trimmedPoints_;;
		vector<std::tuple<float, int, cv::Point2f>> prevPoints_;
		vector<std::tuple<float, int, cv::Point2f>> newPoints_;
		vector<cv::Point2f> upTrimmedPoints_, upNextPoints_;
		std::vector<uchar> status_;
		std::vector<float> err_;
	} temp_;

	std::random_device rd_;
	std::mt19937 rng_;
public:
	SparseOpticalFlow() : rng_(rd_()) {

	}

	//Visualize the sparse optical flow
	void visualize(const cv::UMat &prevGrey, const cv::UMat &nextGrey, const vector<cv::Point2f> &detectedPoints, const float& maxStroke, const size_t& maxPoints, const float& pointLoss, const float& fgScale, cv::Scalar_<float> effectColor) {
		//less then 5 points is a degenerate case (e.g. the corners of a video frame)
	    if (detectedPoints.size() > 4) {
	        cv::convexHull(detectedPoints, temp_.hull_);
	        float area = cv::contourArea(temp_.hull_);
	        //make sure the area of the point cloud is positive
	        if (area > 0) {
	            float density = (detectedPoints.size() / area);
	            //stroke size is biased by the area of the point cloud
	            float strokeSize = maxStroke * pow(area / (nextGrey.cols * nextGrey.rows), 0.33f);
	            //max points is biased by the densitiy of the point cloud
	            size_t currentMaxPoints = ceil(density * maxPoints);

	            //lose a number of random points specified by pointLossPercent
	            std::shuffle(temp_.prevPoints_.begin(), temp_.prevPoints_.end(), rng_);
	            temp_.prevPoints_.resize(ceil(temp_.prevPoints_.size() * (1.0f - (pointLoss / 100.0f))));
	            temp_.trimmedPoints_.clear();
	            for(size_t i = 0; i < temp_.prevPoints_.size(); ++i) {
	            	temp_.trimmedPoints_.push_back(std::get<2>(temp_.prevPoints_[i]));
	            }

	            //calculate how many newly detected points to add
	            size_t copyn = std::min(detectedPoints.size(), (size_t(std::ceil(currentMaxPoints)) - temp_.trimmedPoints_.size()));
	            if (temp_.trimmedPoints_.size() < currentMaxPoints) {
	                std::copy(detectedPoints.begin(), detectedPoints.begin() + copyn, std::back_inserter(temp_.trimmedPoints_));
	            }

	            //calculate the sparse optical flow
	            cv::calcOpticalFlowPyrLK(prevGrey, nextGrey, temp_.trimmedPoints_, temp_.nextPoints_, temp_.status_, temp_.err_);
	            temp_.newPoints_.clear();
	            if (temp_.trimmedPoints_.size() > 1 && temp_.nextPoints_.size() > 1) {
	                //scale the points to original size
	            	temp_.upNextPoints_.clear();
	            	temp_.upTrimmedPoints_.clear();
	                for (cv::Point2f pt : temp_.trimmedPoints_) {
	                	temp_.upTrimmedPoints_.push_back(pt /= fgScale);
	                }

	                for (cv::Point2f pt : temp_.nextPoints_) {
	                	temp_.upNextPoints_.push_back(pt /= fgScale);
	                }

	                for (size_t i = 0; i < temp_.trimmedPoints_.size(); i++) {
	                    if (temp_.status_[i] == 1 //point was found in prev and new set
	                            && temp_.err_[i] < (1.0 / density) //with a higher density be more sensitive to the feature error
	                            && temp_.upNextPoints_[i].y >= 0 && temp_.upNextPoints_[i].x >= 0 //check bounds
	                            && temp_.upNextPoints_[i].y < nextGrey.rows / fgScale && temp_.upNextPoints_[i].x < nextGrey.cols / fgScale //check bounds
	                            ) {
	                        float len = hypot(fabs(temp_.upTrimmedPoints_[i].x - temp_.upNextPoints_[i].x), fabs(temp_.upTrimmedPoints_[i].y - temp_.upNextPoints_[i].y));
	                        if(len > strokeSize) {
	                        	temp_.newPoints_.push_back({len, i, temp_.nextPoints_[i]});
	                        }
	                    }
	                }
//	                std::cerr << "new points:" << temp_.newPoints_.size() << std::endl;
	                if(temp_.newPoints_.empty())
	                	return;
	                float total = 0;
	                float mean = 0;
	                for (size_t i = 0; i < temp_.newPoints_.size(); i++) {
	                	total += std::get<0>(temp_.newPoints_[i]);
	                }

	                mean = total / temp_.newPoints_.size();

	                using namespace cv::v4d::nvg;
	                //start drawing
	                beginPath();
	                strokeWidth(strokeSize);
	                strokeColor(cv::Scalar(effectColor[2], effectColor[1], effectColor[0], effectColor[3]) * 255.0);

	                for (size_t i = 0; i < temp_.newPoints_.size(); i++) {
	                	size_t idx = std::get<1>(temp_.newPoints_[i]);
	                	float len = std::get<0>(temp_.newPoints_[i]);
	                	if(len < mean * 2.0) {
	                		moveTo(temp_.upTrimmedPoints_[idx].x, temp_.upTrimmedPoints_[idx].y);
	                		lineTo(temp_.upNextPoints_[idx].x, temp_.upNextPoints_[idx].y);
	                	}
	                }
	                //end drawing
	                stroke();
	            }
	            temp_.prevPoints_ = temp_.newPoints_;
	        }
	    }
	}

};

class OptflowDemoPlan : public Plan {
private:
	constexpr static auto UMAT_CREATE = _OLM_(void, cv::UMat, &cv::UMat::create, cv::Size, int, cv::UMatUsageFlags);
	constexpr static auto UMAT_DIVIDE_= _OL_(void, cv::divide, cv::InputArray, cv::InputArray, cv::OutputArray, double, int);
	constexpr static auto UMAT_COPY_TO_= _OLMC_(void, cv::UMat, &cv::UMat::copyTo, cv::OutputArray);

	static struct Params {
		// Generate the foreground at this scale.
		float fgScale_ = 0.5f;
		// On every frame the foreground loses on brightness. Specifies the loss in percent.
		float fgLoss_ = 20.0f;
		PostProcessor::Modes postProcMode_ = PostProcessor::GLOW;
		// Intensity of glow or bloom defined by kernel size. The default scales with the image diagonal.
		int kernelSize_ = 0;
		//The lightness selection threshold
		int bloomThresh_ = 210;
		//The intensity of the bloom filter
		float bloomGain_ = 3;
		//Convert the background to greyscale
		BackgroundStyle::Modes backgroundMode_ = BackgroundStyle::GREY;
		// Peak thresholds for the scene change detection. Lowering them makes the detection more sensitive but
		// the default should be fine.
		float sceneChangeThresh_ = 0.29f;
		float sceneChangeThreshDiff_ = 0.1f;
		// The theoretical maximum number of points to track which is scaled by the density of detected points
		// and therefor is usually much smaller.
		int maxPoints_ = 300000;
		// How many of the tracked points to lose intentionally, in percent.
		float pointLoss_ = 10;
		// The theoretical maximum size of the drawing stroke which is scaled by the area of the convex hull
		// of tracked points and therefor is usually much smaller.
		int maxStroke_ = 6;
		// Red, green, blue and alpha. All from 0.0f to 1.0f
		cv::Scalar_<float> effectColor_ = {1.0f, 0.5f, 0.0f, 1.0f};
		//display on-screen FPS
		bool showFps_ = true;
		//Stretch frame buffer to window size
		bool stretch_ = true;
		//The post processing mode
	} params_;

	struct Frames {
		//BGRA
		cv::UMat background_, down_;

		//BGR
		cv::UMat result_;
		//GREY
		cv::UMat downPrevGrey_, downNextGrey_, downMotionMaskGrey_;
	} frames_;

	FeaturePoints featurePoints_;
	SceneChange sceneChange_;
	SparseOpticalFlow sparseOptflow_;
	Compositor compositor_;
	vector<cv::Point2f> detectedPoints_;
	inline static cv::UMat foreground_;

	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
	Property<size_t> numWorkers_ = P<size_t>(Global::Keys::WORKERS_STARTED);

    class PrepareMasksPlan : public Plan {
    	struct Temp {
    		cv::UMat srcGray_;
    		cv::UMat srcGrayFloat_;
    		cv::UMat lastMmGray_;
    		cv::UMat mmGray_;
    		cv::UMat mmEqGray_;
    		cv::UMat mmGrayFloat_;
    		cv::UMat mmBlurGray_;
    		cv::UMat mmBlurGrayFloat_;
    	} temp_;

		OptflowDemoPlan::Frames& frames_;
    	cv::Ptr<cv::BackgroundSubtractor> bgSubtractor_;
    	cv::Mat bigElement_;
    	cv::Mat smallElement_;
    	cv::Size sz_;

    	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);

    	void prepareMotionMask(const cv::UMat& srcGray, cv::UMat& motionMaskGrey) {
    		bgSubtractor_->apply(srcGray, temp_.mmGray_);
    		temp_.mmGray_.convertTo(temp_.mmGrayFloat_, CV_64F, 1.0/255.1);
    		cv::pow(temp_.mmGrayFloat_, 8, temp_.mmGrayFloat_);
    		cv::boxFilter(temp_.mmGrayFloat_, temp_.mmBlurGrayFloat_, -1, cv::Size(31, 31), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    		temp_.mmBlurGrayFloat_.convertTo(temp_.mmGray_, CV_8U, 255.0);
    		cv::morphologyEx(temp_.mmGray_, temp_.mmGray_,cv::MORPH_OPEN, bigElement_, cv::Point(bigElement_.cols >> 1, bigElement_.rows >> 1),2, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
//    		cv::boxFilter(temp_.mmGray_, temp_.mmGray_, -1, cv::Size(11, 11), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    		cv::normalize(temp_.mmGray_, temp_.mmGray_,120, 234, cv::NORM_MINMAX);
    		cv::threshold(temp_.mmGray_, temp_.mmGray_,128, 255, cv::THRESH_TOZERO);
    		cv::morphologyEx(temp_.mmGray_, temp_.mmGray_, cv::MORPH_OPEN, smallElement_, cv::Point(smallElement_.cols >> 1, smallElement_.rows >> 1), 7, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
    		if(motionMaskGrey.empty())
    			motionMaskGrey.create(srcGray.size(), srcGray.type());
    		else
    			motionMaskGrey.setTo(cv::Scalar::all(0));
    		cv::equalizeHist(srcGray, temp_.mmEqGray_);
    		temp_.mmEqGray_.copyTo(motionMaskGrey,temp_.mmGray_);
    		cv::threshold(motionMaskGrey, motionMaskGrey, 127, 255, cv::THRESH_BINARY);
    	}
    public:
    	PrepareMasksPlan(OptflowDemoPlan::Frames& frames) : frames_(frames) {
    	}

    	void setup() override {
    		assign(RW(bgSubtractor_), F(cv::createBackgroundSubtractorMOG2, V(100), V(16), V(true)));
    		assign(RW(bigElement_), F(cv::getStructuringElement, V(cv::MORPH_ELLIPSE), V(cv::Size(7, 7)), V(cv::Point(4, 4))));
    		assign(RW(smallElement_), F(cv::getStructuringElement, V(cv::MORPH_ELLIPSE), V(cv::Size(7, 7)), V(cv::Point(4, 4))));
    	}
    	void infer() override {
    		construct(RW(sz_), F(&cv::Rect::width, vp_) * CS(params_.fgScale_), F(&cv::Rect::height, vp_) * CS(params_.fgScale_));
    		plain(cv::resize, R(frames_.background_), RW(frames_.down_), R(sz_), V(0.0), V(0.0), V(cv::INTER_LINEAR));
    		plain(cv::cvtColor, R(frames_.down_), RW(frames_.downNextGrey_), V(cv::COLOR_RGBA2GRAY), V(0), V(cv::ALGO_HINT_DEFAULT));
    		plain(&PrepareMasksPlan::prepareMotionMask, RW(*this), R(frames_.downNextGrey_), RW(frames_.downMotionMaskGrey_));
    	}
    };

    cv::Ptr<PrepareMasksPlan> prepareMasks_;
public:
    OptflowDemoPlan() {
		prepareMasks_ = _sub<PrepareMasksPlan>(this, frames_);
    }

    void gui() override {
		imgui([](Params& params){
	        using namespace ImGui;

	        Begin("Effects");
	        Text("Foreground");
	        SliderFloat("Scale", &params.fgScale_, 0.1f, 4.0f);
	        SliderFloat("Loss", &params.fgLoss_, 0.1f, 99.9f);
	        Text("Background");
	        thread_local const char* bgm_items[4] = {"Grey", "Color", "Value", "Black"};
	        thread_local int* bgm = (int*)&params.backgroundMode_;
	        ListBox("Mode", bgm, bgm_items, 4, 4);
	        Text("Points");
	        SliderInt("Max. Points", &params.maxPoints_, 10, 10000000);
	        SliderFloat("Point Loss", &params.pointLoss_, 0.0f, 100.0f);
	        Text("Optical flow");
	        SliderInt("Max. Stroke Size", &params.maxStroke_, 1, 100);
	        ColorPicker4("Color", params.effectColor_.val);
	        End();

	        Begin("Post Processing");
	        thread_local const char* ppm_items[3] = {"Glow", "Bloom", "None"};
	        thread_local int* ppm = (int*)&params.postProcMode_;
	        ListBox("Effect",ppm, ppm_items, 3, 3);
	        SliderInt("Kernel Size",&params.kernelSize_, 1, 63);
	        SliderFloat("Gain", &params.bloomGain_, 0.1f, 20.0f);
	        End();

	        Begin("Settings");
	        Text("Scene Change Detection");
	        SliderFloat("Threshold", &params.sceneChangeThresh_, 0.1f, 1.0f);
	        SliderFloat("Threshold Diff", &params.sceneChangeThreshDiff_, 0.1f, 1.0f);
	        End();

			Begin("Window");
			if(Checkbox("Show FPS", &params.showFps_)) {
//				win->setShowFPS(params.showFps_);
			}
			if(Checkbox("Stretch", &params.stretch_)) {
//				win->setStretching(params.stretch_);
			}

			if(Button("Fullscreen")) {
//				win->setFullscreen(!win->isFullscreen());
			};

			if(Button("Offscreen")) {
//				win->setVisible(!win->isVisible());
			};

			End();
	    }, params_);
	}

    void setup() override {
    	construct(RW(featurePoints_), F(cv::FastFeatureDetector::create, V(10), V(false), V(cv::FastFeatureDetector::TYPE_9_16)));
    	branch(BranchType::ONCE, always_)
			->assign(RWS(params_.kernelSize_),
					F(sqrt, F(&cv::Rect::width, vp_) * F(&cv::Rect::height, vp_))
					/ V(400.0)
			)
			->assign(RWS(params_.effectColor_[3]),RW(params_.effectColor_[3]) / F(pow, numWorkers_, V(0.5) / numWorkers_))
			->plain(UMAT_CREATE,
						RWS(foreground_),
						F(&cv::Rect::size, vp_),
						V(CV_8UC4),
						V(cv::USAGE_DEFAULT)
			)
		->endBranch();
    	subSetup(prepareMasks_);
	}

	void infer() override {
		set(V4D::Keys::STRETCHING, CS(params_.stretch_));
		capture();

		fb(UMAT_COPY_TO_, RW(frames_.background_));
		subInfer(prepareMasks_);

		plain(&FeaturePoints::detect, RW(featurePoints_),
				R(frames_.downMotionMaskGrey_),
				RW(detectedPoints_)
		);

		fb<1>(UMAT_COPY_TO_, RWS(foreground_));
		branch(!F(&SceneChange::detect, RW(sceneChange_),
				R(frames_.downMotionMaskGrey_),
				CS(params_.sceneChangeThresh_),
				CS(params_.sceneChangeThreshDiff_)
			)
		)
			->branch(!F(&cv::UMat::empty, R(frames_.downPrevGrey_)))
				->nvg(&SparseOpticalFlow::visualize, RW(sparseOptflow_),
						R(frames_.downPrevGrey_),
						R(frames_.downNextGrey_),
						R(detectedPoints_),
						CS(params_.maxStroke_),
						CS(params_.maxPoints_),
						CS(params_.pointLoss_),
						CS(params_.fgScale_),
						CS(params_.effectColor_)
				)
				->fb(UMAT_COPY_TO_, RWS(foreground_))
			->endBranch()
		->endBranch();

		fb<3>(&Compositor::perform, RW(compositor_),
							R(frames_.background_),
							RWS(foreground_),
							CS(params_.backgroundMode_),
							CS(params_.postProcMode_),
							CS(params_.kernelSize_),
							CS(params_.bloomThresh_),
							CS(params_.bloomGain_),
							CS(params_.fgLoss_),
							numWorkers_
		);
		plain(UMAT_COPY_TO_, R(frames_.downNextGrey_), RW(frames_.downPrevGrey_));
	}
};

OptflowDemoPlan::Params OptflowDemoPlan::params_;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: optflow-demo <input-video-file>" << endl;
        exit(1);
    }

    cv::Rect viewport(0, 0, 1920, 1080);
	cv::Ptr<V4D> runtime = V4D::init(viewport, "Sparse Optical Flow Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
	auto src = Source::make(runtime, argv[1]);
	runtime->setSource(src);
	Plan::run<OptflowDemoPlan>(2);

    return 0;
}
