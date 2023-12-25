// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

using std::cerr;
using std::endl;

using namespace cv::v4d;

class VideoDemoPlan: public Plan {
private:
	/* Scene constants */
	constexpr static GLuint TRIANGLES_ = 12;
	constexpr static GLuint VERTICES_INDEX_ = 0;
	constexpr static GLuint COLOR_INDEX_ = 1;

    constexpr static float VERTICES_[24] = {
            // Front face
            0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5,

            // Back face
            0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, };

    constexpr static float VERTEX_COLORS[24] = { 1.0, 0.4, 0.6, 1.0, 0.9, 0.2, 0.7, 0.3, 0.8, 0.5, 0.3, 1.0,

    0.2, 0.6, 1.0, 0.6, 1.0, 0.4, 0.6, 0.8, 0.8, 0.4, 0.8, 0.8, };

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
            5, 1, 0, 0, 4, 5, };

	/* OpenGL handles */
    struct Handles {
		GLuint vao_ = 0;
		GLuint program_ = 0;
		GLuint uniform_ = 0;
		GLuint trianglesEbo_ = 0;
		GLuint verticesVbo_ = 0;
		GLuint colorsVbo_ = 0;
    } handles_;

	static GLuint load_shaders() {
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

        unsigned int handles[3];
        cv::v4d::init_shaders(handles, vert.c_str(), frag.c_str(), "fragColor");
        return handles[0];
	}

	static void init_scene(Handles& handles) {
	    glEnable (GL_DEPTH_TEST);

	    glGenVertexArrays(1, &handles.vao_);
	    glBindVertexArray(handles.vao_);

	    glGenBuffers(1, &handles.trianglesEbo_);
	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handles.trianglesEbo_);
	    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof TRIANGLE_INDICES_, TRIANGLE_INDICES_,
	            GL_STATIC_DRAW);

	    glGenBuffers(1, &handles.verticesVbo_);
	    glBindBuffer(GL_ARRAY_BUFFER, handles.verticesVbo_);
	    glBufferData(GL_ARRAY_BUFFER, sizeof VERTICES_, VERTICES_, GL_STATIC_DRAW);

	    glVertexAttribPointer(VERTICES_INDEX_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	    glEnableVertexAttribArray(VERTICES_INDEX_);

	    glGenBuffers(1, &handles.colorsVbo_);
	    glBindBuffer(GL_ARRAY_BUFFER, handles.colorsVbo_);
	    glBufferData(GL_ARRAY_BUFFER, sizeof VERTEX_COLORS, VERTEX_COLORS, GL_STATIC_DRAW);

	    glVertexAttribPointer(COLOR_INDEX_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	    glEnableVertexAttribArray(COLOR_INDEX_);

	    glBindVertexArray(0);
	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	    glBindBuffer(GL_ARRAY_BUFFER, 0);

	    handles.program_ = load_shaders();
	    handles.uniform_ = glGetUniformLocation(handles.program_, "transform");
	}

	static void destroy_scene(const Handles& handles) {
		glDeleteProgram(handles.program_);
		glDeleteBuffers(1, &handles.colorsVbo_);
		glDeleteBuffers(1, &handles.verticesVbo_);
		glDeleteBuffers(1, &handles.trianglesEbo_);
		glDeleteVertexArrays(1, &handles.vao_);
	}

	static void render_scene(const Handles& handles) {
	    glUseProgram(handles.program_);

	    float angle = fmod(double(cv::getTickCount()) / double(cv::getTickFrequency()), 2 * M_PI);
	    float scale = 0.25;

	    cv::Matx44f scaleMat(scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0,
	            0.0, 1.0);

	    cv::Matx44f rotXMat(1.0, 0.0, 0.0, 0.0, 0.0, cos(angle), -sin(angle), 0.0, 0.0, sin(angle),
	            cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0);

	    cv::Matx44f rotYMat(cos(angle), 0.0, sin(angle), 0.0, 0.0, 1.0, 0.0, 0.0, -sin(angle), 0.0,
	            cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0);

	    cv::Matx44f rotZMat(cos(angle), -sin(angle), 0.0, 0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0,
	            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

	    cv::Matx44f transform = scaleMat * rotXMat * rotYMat * rotZMat;
	    glUniformMatrix4fv(handles.uniform_, 1, GL_FALSE, transform.val);
	    glBindVertexArray(handles.vao_);
	    glDrawElements(GL_TRIANGLES, TRIANGLES_ * 3, GL_UNSIGNED_SHORT, NULL);
	}
public:
	void setup() override {
		gl([](Handles& handles) {
			init_scene(handles);
		}, RW(handles_));
	}

	void infer() override {
		gl([]() {
			glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		});

		capture();

		gl([](const Handles& handles) {
			render_scene(handles);
		}, R(handles_));

		write();
	}

	void teardown() override {
		gl([](const Handles& handles) {
			destroy_scene(handles);
		}, R(handles_));
	}
};


int main(int argc, char** argv) {
	if (argc != 2) {
        cerr << "Usage: video-demo <video-file>" << endl;
        exit(1);
    }

	cv::Rect viewport(0,0,1280,720);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Video Demo", AllocateFlags::IMGUI);
    auto src = Source::make(runtime, argv[1]);
    auto sink = Sink::make(runtime, "video-demo.mkv", src->fps(), viewport.size());
    runtime->setSource(src);
    runtime->setSink(sink);
    Plan::run<VideoDemoPlan>(0);

    return 0;
}
