// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

#include "cubescene.hpp"

using namespace cv::v4d;

class ManyCubesDemoPlan : public Plan {
	constexpr static size_t NUMBER_OF_CONTEXTS_ = 10;

	CubeScene scene_;
	size_t currentGlCtx_ = 0;
	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void setup() override {
		set(_(V4D::Keys::CLEAR_COLOR, V(cv::Scalar(202, 61, 51, 255.0))));

		for(size_t i = 0; i < NUMBER_OF_CONTEXTS_; ++i) {
			gl<-1>(V(i), &CubeScene::init, RW(scene_));
		}
	}

	void infer() override {
		//Render using multiple OpenGL contexts
		clear();
		for(size_t i = 0; i < NUMBER_OF_CONTEXTS_; ++i) {
			gl<-1>(V(i),
				&CubeScene::render, R(scene_),
					V(sin((double(i) / NUMBER_OF_CONTEXTS_) * 2.0 * CV_PI) / 1.5),
					V(cos((double(i) / NUMBER_OF_CONTEXTS_) * 2.0 * CV_PI) / 1.5));
		}
	}

	void teardown() override {
		for(size_t i = 0; i < NUMBER_OF_CONTEXTS_; ++i) {
			gl<-1>(V(i), &CubeScene::destroy, R(scene_));
		}
	}
};

int main() {
	cv::Rect viewport(0, 0, 1920, 1080);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Many Cubes Demo", AllocateFlags::IMGUI);
    Plan::run<ManyCubesDemoPlan>(2);

    return 0;
}
