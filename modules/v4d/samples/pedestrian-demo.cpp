// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>

#include <string>

using std::vector;
using std::string;

using namespace cv::v4d;

class PedestrianDemoPlan : public Plan {
public:
	using Plan::Plan;
private:
	cv::Size downSize_;
	cv::Size_<float> scale_;

	struct Frames {
		//BGRA
		cv::UMat background_;
    	//RGB
    	cv::UMat videoFrame_, videoFrameDown_;
    	//GREY
    	cv::UMat videoFrameDownGrey_;
	} frames_;

    struct Detection {
		//detected pedestrian locations rectangles
		std::vector<cv::Rect> locations_;
		//detected pedestrian locations as boxes
		vector<vector<double>> boxes_;
		//probability of detected object being a pedestrian - currently always set to 1.0
		vector<double> probs_;
		//Faster tracking parameters
		cv::TrackerKCF::Params params_;
		//KCF tracker used instead of continous detection
		cv::Ptr<cv::Tracker> tracker_;
		//initialize tracker only once
		bool trackerInit_ = false;
		//If tracking fails re-detect
		bool redetect_ = true;
		//Descriptor used for pedestrian detection
		cv::HOGDescriptor hog_;
    } detection_;

    inline static cv::Rect tracked_ = cv::Rect(0,0,0,0);

	constexpr static auto doRedect_ = [](const Detection& detection){ return !detection.trackerInit_ || detection.redetect_; };

	//adapted from cv::dnn_objdetect::InferBbox
	static inline bool pair_comparator(std::pair<double, size_t> l1, std::pair<double, size_t> l2) {
	    return l1.first > l2.first;
	}

	//adapted from cv::dnn_objdetect::InferBbox
	static void intersection_over_union(std::vector<std::vector<double> > *boxes, std::vector<double> *base_box, std::vector<double> *iou) {
	    double g_xmin = (*base_box)[0];
	    double g_ymin = (*base_box)[1];
	    double g_xmax = (*base_box)[2];
	    double g_ymax = (*base_box)[3];
	    double base_box_w = g_xmax - g_xmin;
	    double base_box_h = g_ymax - g_ymin;
	    for (size_t b = 0; b < (*boxes).size(); ++b) {
	        double xmin = std::max((*boxes)[b][0], g_xmin);
	        double ymin = std::max((*boxes)[b][1], g_ymin);
	        double xmax = std::min((*boxes)[b][2], g_xmax);
	        double ymax = std::min((*boxes)[b][3], g_ymax);

	        // Intersection
	        double w = std::max(static_cast<double>(0.0), xmax - xmin);
	        double h = std::max(static_cast<double>(0.0), ymax - ymin);
	        // Union
	        double test_box_w = (*boxes)[b][2] - (*boxes)[b][0];
	        double test_box_h = (*boxes)[b][3] - (*boxes)[b][1];

	        double inter_ = w * h;
	        double union_ = test_box_h * test_box_w + base_box_h * base_box_w - inter_;
	        (*iou)[b] = inter_ / (union_ + 1e-7);
	    }
	}

	//adapted from cv::dnn_objdetect::InferBbox
	static std::vector<bool> non_maximal_suppression(std::vector<std::vector<double> > *boxes, std::vector<double> *probs, const double threshold = 0.1) {
	    std::vector<bool> keep(((*probs).size()));
	    std::fill(keep.begin(), keep.end(), true);
	    std::vector<size_t> prob_args_sorted((*probs).size());

	    std::vector<std::pair<double, size_t> > temp_sort((*probs).size());
	    for (size_t tidx = 0; tidx < (*probs).size(); ++tidx) {
	        temp_sort[tidx] = std::make_pair((*probs)[tidx], static_cast<size_t>(tidx));
	    }
	    std::sort(temp_sort.begin(), temp_sort.end(), pair_comparator);

	    for (size_t idx = 0; idx < temp_sort.size(); ++idx) {
	        prob_args_sorted[idx] = temp_sort[idx].second;
	    }

	    for (std::vector<size_t>::iterator itr = prob_args_sorted.begin(); itr != prob_args_sorted.end() - 1; ++itr) {
	        size_t idx = itr - prob_args_sorted.begin();
	        std::vector<double> iou_(prob_args_sorted.size() - idx - 1);
	        std::vector<std::vector<double> > temp_boxes(iou_.size());
	        for (size_t bb = 0; bb < temp_boxes.size(); ++bb) {
	            std::vector<double> temp_box(4);
	            for (size_t b = 0; b < 4; ++b) {
	                temp_box[b] = (*boxes)[prob_args_sorted[idx + bb + 1]][b];
	            }
	            temp_boxes[bb] = temp_box;
	        }
	        intersection_over_union(&temp_boxes, &(*boxes)[prob_args_sorted[idx]], &iou_);
	        for (std::vector<double>::iterator _itr = iou_.begin(); _itr != iou_.end(); ++_itr) {
	            size_t iou_idx = _itr - iou_.begin();
	            if (*_itr > threshold) {
	                keep[prob_args_sorted[idx + iou_idx + 1]] = false;
	            }
	        }
	    }
	    return keep;
	}
public:
    PedestrianDemoPlan(const cv::Rect& viewport) : Plan(viewport) {
    	Global::registerShared(tracked_);
    	int w = size().width;
    	int h = size().height;
    	downSize_ = { 640 , 360 };
    	scale_ = { float(w) / downSize_.width, float(h) / downSize_.height };
    }

    void setup(cv::Ptr<V4D> window) override {
    	window->plain([](Detection& detection){
    		detection.params_.desc_pca = cv::TrackerKCF::GRAY;
    		detection.params_.compress_feature = false;
    		detection.params_.compressed_size = 1;
    		detection.tracker_ = cv::TrackerKCF::create(detection.params_);
    		detection.hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
		}, RW(detection_));
	}

	void infer(cv::Ptr<V4D> window) override {
		window->capture();

		window->fb([](const cv::UMat& framebuffer, cv::UMat& videoFrame){
			//copy video frame
			cvtColor(framebuffer,videoFrame,cv::COLOR_BGRA2RGB);
		}, RW(frames_.videoFrame_));

		window->plain([](const cv::Size& downSize, Frames &frames){
			cv::resize(frames.videoFrame_, frames.videoFrameDown_, downSize);
			cv::cvtColor(frames.videoFrameDown_, frames.videoFrameDownGrey_, cv::COLOR_RGB2GRAY);
			cv::cvtColor(frames.videoFrame_, frames.background_, cv::COLOR_RGB2BGRA);
		}, R(downSize_), RW(frames_));

		//Try to track the pedestrian (if we currently are tracking one), else re-detect using HOG descriptor
		window->branch(doRedect_, R(detection_))
			->plain([](const cv::UMat& videoFrameDownGrey, Detection& detection, cv::Rect& tracked) {
				//Detect pedestrians
				detection.hog_.detectMultiScale(videoFrameDownGrey, detection.locations_, 0, cv::Size(), cv::Size(), 1.15, 2.0, true);
				if (!detection.locations_.empty()) {
					cerr << "detected" << endl;
					detection.boxes_.clear();
					detection.probs_.clear();
					//collect all found boxes
					for (const auto &rect : detection.locations_) {
						detection.boxes_.push_back( { double(rect.x), double(rect.y), double(rect.x + rect.width), double(rect.y + rect.height) });
						detection.probs_.push_back(1.0);
					}

					//use nms to filter overlapping boxes (https://medium.com/analytics-vidhya/non-max-suppression-nms-6623e6572536)
					vector<bool> keep = non_maximal_suppression(&detection.boxes_, &detection.probs_, 0.1);
					for (size_t i = 0; i < keep.size(); ++i) {
						if (keep[i]) {
							//only track the first pedestrian found
							if(tracked.width == 0 || tracked.height == 0) {
								tracked = detection.locations_[i];
							} else {
								tracked.x = (detection.locations_[i].x + tracked.x) / 2.0;
								tracked.y = (detection.locations_[i].y + tracked.y) / 2.0;
								tracked.width = (detection.locations_[i].width + tracked.width) / 2.0;
								tracked.height = (detection.locations_[i].height + tracked.height) / 2.0;
							}
							detection.redetect_ = false;
							break;
						}
					}

					if(!detection.trackerInit_){
						//initialize the tracker once
						detection.tracker_->init(videoFrameDownGrey, tracked);
						detection.trackerInit_ = true;
					}
				}
				cerr << "redectend: " << tracked << endl;
			}, R(frames_.videoFrameDownGrey_), RW(detection_), RW_C(tracked_))
		->elseBranch()
			->plain([](const cv::UMat& videoFrameDownGrey, Detection& detection, cv::Rect& tracked) {
				cv::Rect newTracked = tracked;
				if(newTracked.width == 0 || newTracked.height == 0 || !detection.tracker_->update(videoFrameDownGrey, newTracked)) {
					detection.redetect_ = true;
				} else {
					tracked.x = (newTracked.x + tracked.x) / 2.0;
					tracked.y = (newTracked.y + tracked.y) / 2.0;
					tracked.width = (newTracked.width + tracked.width) / 2.0;
					tracked.height = (newTracked.height+ tracked.height) / 2.0;
				}
			}, R(frames_.videoFrameDownGrey_), RW(detection_), RW_C(tracked_))
		->endBranch();

		//Draw an ellipse around the tracked pedestrian
		window->nvg([](const cv::Size& sz, const cv::Size_<float>& scale, const cv::Rect tracked) {
			using namespace cv::v4d::nvg;
			float width = tracked.width * scale.width;
			float height = tracked.height * scale.height;
			float cx = (scale.width * tracked.x + (width / 2.0));
			float cy = (scale.height * tracked.y + ((height) / 2.0));

			clear();
			beginPath();
			strokeWidth(std::fmax(5.0, sz.width / 960.0));
			strokeColor(cv::v4d::colorConvert(cv::Scalar(0, 127, 255, 200), cv::COLOR_HLS2BGR));
			ellipse(cx, cy, (width), (height));
			stroke();
		}, R(size()), R(scale_), R_C(tracked_));

		//Put it all together
		window->fb([](cv::UMat& framebuffer, const cv::UMat& background) {
	        cv::add(background, framebuffer, framebuffer);
		}, R(frames_.background_));
	}
};


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: pedestrian-demo <video-input>" << endl;
        exit(1);
    }

    cv::Rect viewport(0, 0, 1280, 720);
    cv::Ptr<V4D> window = V4D::make(viewport.size(), "Pedestrian Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
    auto src = Source::make(window, argv[1]);
    window->setSource(src);
    window->run<PedestrianDemoPlan>(0, viewport);
    return 0;
}
