// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

#include "cubescene.hpp"

using namespace cv::v4d;

class CubeDemoPlan : public Plan {
	CubeScene scene_;
public:
	void setup() override {
		set(_(V4D::Keys::CLEAR_COLOR, V(cv::Scalar(202, 61, 51, 255.0))));
		gl(&CubeScene::init, RW(scene_));
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
