// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

#include "cubescene.hpp"

using namespace cv::v4d;

class CubeDemoPlan : public Plan {
	constexpr static size_t WIDTH_ = 1920;
	constexpr static size_t HEIGHT_ = 1080;
	CubeScene scene_;
public:
	void setup() override {
		set(_(V4D::Keys::CLEAR_COLOR, V(cv::Scalar(102, 61, 51, 255))));
		gl(&CubeScene::init, RW(scene_));
		clear();
		gl(glEnable, V(GL_SCISSOR_TEST));
		gl(glScissor, V(760), V(400), V(400), V(280));
	}

	void infer() override {
		clear();
		gl(&CubeScene::render, R(scene_), V(0.0), V(0.0));
	}

	void teardown() override {
		gl(&CubeScene::destroy, R(scene_));
	}
};

int main() {
	cv::Rect viewport(0, 0, 1920, 1080);
	cv::Ptr<V4D> runtime = V4D::init(viewport, "Cube Demo", AllocateFlags::IMGUI);
	Plan::run<CubeDemoPlan>(2);

	return 0;
}
