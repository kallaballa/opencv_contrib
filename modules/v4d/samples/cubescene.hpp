// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SAMPLES_OPENCV_V4D_CUBESCENE_HPP_
#define SAMPLES_OPENCV_V4D_CUBESCENE_HPP_

#include <opencv2/v4d/v4d.hpp>
//adapted from https://gitlab.com/wikibooks-opengl/modern-tutorials/-/blob/master/tut05_cube/cube.cpp

using namespace cv::v4d;

class CubeScene {
	/* OpenGL constants */
	constexpr static GLuint TRIANGLES_ = 12;
	constexpr static GLuint VERTICES_INDEX_ = 0;
	constexpr static GLuint COLOR_INDEX_ = 1;

	//Cube vertices, colors and indices
	constexpr static float VERTICES_[24] = {
			// Front face
			0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5,
			// Back face
			0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5
	};

	constexpr static float VERTEX_COLORS_[24] = {
			1.0, 0.4, 0.6, 1.0, 0.9, 0.2, 0.7, 0.3, 0.8, 0.5, 0.3, 1.0,
			0.2, 0.6, 1.0, 0.6, 1.0, 0.4, 0.6, 0.8, 0.8, 0.4, 0.8, 0.8
	};

	constexpr static unsigned short TRIANGLE_INDICES_[36] = {
			// Front
			0, 1, 2, 2, 3, 0,

			// Right
			0, 3, 7, 7, 4, 0,

			// Bottom
			2, 6, 7, 7, 3, 2,

			// Left
			1, 5, 6, 6, 2, 1,

			// Back
			4, 7, 6, 6, 5, 4,

			// Top
			5, 1, 0, 0, 4, 5
	};

	struct Handles {
		GLuint vao_ = 0;
		GLuint program_ = 0;
		GLuint uniform_= 0;
		GLuint trianglesEbo_ = 0;
		GLuint verticesVbo_ = 0;
		GLuint colorsVbo_ = 0;
	} handles_;

	//Simple transform & pass-through shaders
	static GLuint load_shader() {
		//Shader versions "330" and "300 es" are very similar.
		//If you are careful you can write the same code for both versions.
	#if !defined(OPENCV_V4D_USE_ES3)
		const string shaderVersion = "330";
	#else
		const string shaderVersion = "300 es";
	#endif

		const string vert =
				"    #version " + shaderVersion
						+ R"(
		precision lowp float;
		layout(location = 0) in vec3 pos;
		layout(location = 1) in vec3 vertex_color;
		
		uniform mat4 transform;
		
		out vec3 color;
		void main() {
		  gl_Position = transform * vec4(pos, 1.0);
		  color = vertex_color;
		}
	)";

		const string frag =
				"    #version " + shaderVersion
						+ R"(
		precision lowp float;
		in vec3 color;
		
		out vec4 frag_color;
		
		void main() {
		  frag_color = vec4(color, 1.0);
		}
	)";

		//Initialize the shaders and returns the program
		unsigned int handles[3];
		cv::v4d::init_shaders(handles, vert.c_str(), frag.c_str(), "fragColor");
		return handles[0];
	}

public:
	//Initializes objects, buffers, shaders and uniforms
	void init() {
		glEnable (GL_DEPTH_TEST);

		glGenVertexArrays(1, &handles_.vao_);
		glBindVertexArray(handles_.vao_);

		glGenBuffers(1, &handles_.trianglesEbo_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handles_.trianglesEbo_);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof TRIANGLE_INDICES_, TRIANGLE_INDICES_,
				GL_STATIC_DRAW);
		glGenBuffers(1, &handles_.verticesVbo_);
		glBindBuffer(GL_ARRAY_BUFFER, handles_.verticesVbo_);
		glBufferData(GL_ARRAY_BUFFER, sizeof VERTICES_, VERTICES_, GL_STATIC_DRAW);

		glVertexAttribPointer(VERTICES_INDEX_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(VERTICES_INDEX_);
		glGenBuffers(1, &handles_.colorsVbo_);
		glBindBuffer(GL_ARRAY_BUFFER, handles_.colorsVbo_);
		glBufferData(GL_ARRAY_BUFFER, sizeof VERTEX_COLORS_, VERTEX_COLORS_, GL_STATIC_DRAW);

		glVertexAttribPointer(COLOR_INDEX_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(COLOR_INDEX_);

		glBindVertexArray(0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		handles_.program_ = load_shader();
		handles_.uniform_ = glGetUniformLocation(handles_.program_, "transform");
		glClearColor(0.2, 0.24, 0.4, 1);
	}

	//Renders a rotating rainbow-colored cube on a blueish background
	void render(const bool clearColorBuffer) const {
		glClear((clearColorBuffer ? GL_COLOR_BUFFER_BIT : 0) | GL_DEPTH_BUFFER_BIT);

		//Use the prepared shader program
		glUseProgram(handles_.program_);

		//Scale and rotate the cube depending on the current time.
		float angle = fmod(
				double(cv::getTickCount()) / double(cv::getTickFrequency()),
				2 * M_PI);
		float scale = 0.25;

		cv::Matx44f scaleMat(scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0,
				scale, 0.0, 0.0, 0.0, 0.0, 1.0);

		cv::Matx44f rotXMat(1.0, 0.0, 0.0, 0.0, 0.0, cos(angle), -sin(angle), 0.0,
				0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0);

		cv::Matx44f rotYMat(cos(angle), 0.0, sin(angle), 0.0, 0.0, 1.0, 0.0, 0.0,
				-sin(angle), 0.0, cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0);

		cv::Matx44f rotZMat(cos(angle), -sin(angle), 0.0, 0.0, sin(angle),
				cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

		//calculate the transform
		cv::Matx44f transform = scaleMat * rotXMat * rotYMat * rotZMat;
		//set the corresponding uniform
		glUniformMatrix4fv(handles_.uniform_, 1, GL_FALSE, transform.val);
		//Bind the prepared vertex array object
		glBindVertexArray(handles_.vao_);
		//Draw
		glDrawElements(GL_TRIANGLES, TRIANGLES_ * 3, GL_UNSIGNED_SHORT, NULL);
	}

	void destroy() const {
		glDeleteProgram(handles_.program_);
		glDeleteBuffers(1, &handles_.colorsVbo_);
		glDeleteBuffers(1, &handles_.verticesVbo_);
		glDeleteBuffers(1, &handles_.trianglesEbo_);
		glDeleteVertexArrays(1, &handles_.vao_);
	}

};

#endif //SAMPLES_OPENCV_V4D_CUBESCENE_HPP_
