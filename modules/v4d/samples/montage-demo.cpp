// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

int v4d_cube_main();
int v4d_many_cubes_main();
int v4d_video_main(int argc, char **argv);
int v4d_nanovg_main(int argc, char **argv);
int v4d_shader_main(int argc, char **argv);
int v4d_font_main();
int v4d_pedestrian_main(int argc, char **argv);
int v4d_optflow_main(int argc, char **argv);
int v4d_beauty_main(int argc, char **argv);
#define main v4d_cube_main
#include "cube-demo.cpp"
#undef main
#define main v4d_many_cubes_main
#include "many_cubes-demo.cpp"
#undef main
#define main v4d_video_main
#include "video-demo.cpp"
#undef main
#define main v4d_nanovg_main
#include "nanovg-demo.cpp"
#undef main
#define main v4d_shader_main
#include "shader-demo.cpp"
#undef main
#define main v4d_font_main
#include "font-demo.cpp"
#undef main
#define main v4d_pedestrian_main
#include "pedestrian-demo.cpp"
#undef main
#define main v4d_optflow_main
#include "optflow-demo.cpp"
#undef main
#define main v4d_beauty_main
#include "beauty-demo.cpp"
#undef main

class BlankPlan : public Plan {
public:
	void infer() override {
		clear();
	}
};

using namespace cv::v4d::event;
class MontageDemoPlan : public Plan {
	using K = V4D::Keys;
	const bool highresMode_;
	const cv::Size tiling_ = cv::Size(3, 3);
	cv::Size tileSz_;
	cv::Rect tileViewport_;
	std::vector<cv::Rect> targetViewports_;
	std::vector<cv::Ptr<Plan>> plans_;
	std::vector<cv::Ptr<Plan>> highResPlans_;

	Property<cv::Size> winSz_ = P<cv::Size>(K::WINDOW_SIZE);
	Property<cv::Rect> initVp_ = P<cv::Rect>(K::INIT_VIEWPORT);

	Event<Mouse> releaseLeft = E<Mouse>(Mouse::RELEASE, Mouse::LEFT);
	Event<Mouse> releaseRight = E<Mouse>(Mouse::RELEASE, Mouse::RIGHT);

	struct Frames {
		std::vector<cv::UMat> results_;
		cv::UMat captured;
	} frames_;

	struct State {
		int32_t lastZoomed_ = -1;
		int32_t zoomed_ = -1;
	};

	static State state_;
	string id_;
public:
	MontageDemoPlan(const cv::Rect& vp, const bool& highresMode) : highresMode_(highresMode) {
		tileSz_ = cv::Size(vp.width / tiling_.width, vp.height / tiling_.height);
		if(highresMode_)
			tileViewport_ = vp;
		else
			tileViewport_ = cv::Rect(0, 0, tileSz_.width, tileSz_.height);

		plans_ = {
				_sub<CubeDemoPlan>(this),
				_sub<ManyCubesDemoPlan>(this),
				_sub<VideoDemoPlan>(this),
				_sub<NanoVGDemoPlan>(this),
				_sub<ShaderDemoPlan>(this, 30),
				_sub<FontDemoPlan>(this),
				_sub<BlankPlan>(this),
//				_sub<PedestrianDemoPlan>(this),
				_sub<BeautyDemoPlan>(this),
				_sub<BlankPlan>(this),
//				_sub<OptflowDemoPlan>(this)
			};

		CV_Assert(tiling_.width * tiling_.height == plans_.size());
		frames_.results_.resize(plans_.size());
		targetViewports_.resize(plans_.size());

		const int w = tileSz_.width;
		const int h = tileSz_.height;

		for(size_t x = 0; x < 3; ++x) {
			for(size_t y = 0; y < 3; ++y) {
				targetViewports_[x * 3 + y] = cv::Rect(w * x, h * y, w, h);
			}
		}
	}

	void setup() override {
		set(K::VIEWPORT, initVp_);
		for(auto plan : plans_) {
			subSetup(plan);
		}


		for(auto hrPlan : highResPlans_) {
			subSetup(hrPlan);
		}
	}

	void infer() override {
		set(K::VIEWPORT, initVp_);

		clear()
		->capture();

		constexpr auto copyToMemFn = static_cast<void (cv::UMat::*)(cv::OutputArray) const>(&cv::UMat::copyTo);
		fb(copyToMemFn, RW(frames_.captured));

		set(K::DISABLE_VIDEO_IO, V(true));
		set(K::DISABLE_INPUT_EVENTS, V(true));
		branch(CS(state_.zoomed_) == V(-1));
		{
			for(size_t i = 0; i < plans_.size(); ++i) {
				auto plan = plans_[i];
				fb<1>(cv::resize, R(frames_.captured), F(&cv::Rect::size, initVp_), V(0), V(0), V(cv::INTER_LINEAR));
				subInfer(plan);
				fb(copyToMemFn, RW(frames_.results_)[V(i)]);
			}
		}
		elseBranch();
		{
			for(size_t i = 0; i < plans_.size(); ++i) {
				auto plan = plans_[i];
				branch(CS(state_.zoomed_) == V(i))
					->fb<1>(copyToMemFn, R(frames_.captured))
					->subInfer(plan)
					->fb(copyToMemFn, RW(frames_.results_)[V(i)])
				->endBranch();
			}
		}
		endBranch();
		set(K::DISABLE_VIDEO_IO, V(false));
		set(K::DISABLE_INPUT_EVENTS, V(false));

		branch(BranchType::SINGLE, always_)
			->plain([](const cv::Rect& initVp, const cv::Size& windowSz, const Mouse::List& reLeft, const Mouse::List& reRight, const std::vector<cv::Rect>& targetViewports, State& state) {
				{
					using namespace cv::v4d::event;
					const double scaleX = double(initVp.width) / windowSz.width;
					const double scaleY = double(initVp.height) / windowSz.height;
					const double scale = std::min(scaleX, scaleY);
					if(state_.zoomed_ > -1) {
						if(!reRight.empty()) {
							state_.zoomed_ = -1;
						}
					} else {
						if(!reLeft.empty()) {
							cv::Point loc = reLeft[0]->position() * scale;
							for(size_t i = 0; i < targetViewports.size(); ++i) {
								if(targetViewports[i].contains(loc)) {
									state.zoomed_ = i;
									break;
								}
							}
						}
					}
				}
			}, initVp_, winSz_, releaseLeft, releaseRight, R(targetViewports_), RWS(state_))
		->endBranch();

		fb([](cv::UMat& framebuffer, const cv::Size& tileSz, const std::vector<cv::Rect>& targetViewports, const State& state, const Frames& frames) {
			if(state.zoomed_ > -1) {
				cv::resize(frames.results_[state.zoomed_], framebuffer, framebuffer.size());
			} else {
				for(size_t i = 0; i < targetViewports.size(); ++i) {
					if(!frames.results_[i].empty()) {
						cv::resize(frames.results_[i], framebuffer(targetViewports[i]), tileSz);
					}
				}
			}
		}, R(tileSz_), R(targetViewports_), CS(state_), R(frames_));

	}


	void teardown() override {
		set(K::VIEWPORT, initVp_);
		for(auto plan : plans_) {
			subTeardown(plan);
		}

		for(auto plan : highResPlans_) {
			subTeardown(plan);
		}
	}
};

MontageDemoPlan::State MontageDemoPlan::state_;

int main(int argc, char** argv) {
	if (argc != 3) {
        cerr << "Usage: montage-demo <video-file> <number of extra workers>" << endl;
        exit(1);
    }
	cv::Rect viewport(0, 0, 1280, 720);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Montage Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI, ConfigFlags::DEFAULT, DebugFlags::DEFAULT);
    auto src = Source::make(runtime, argv[1]);
    runtime->setSource(src);
    Plan::run<MontageDemoPlan>(atoi(argv[2]), viewport, false);

    return 0;
}

