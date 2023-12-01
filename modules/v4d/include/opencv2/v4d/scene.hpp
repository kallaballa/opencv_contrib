// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef MODULES_V4D_SRC_SCENE_HPP_
#define MODULES_V4D_SRC_SCENE_HPP_

#include "v4d.hpp"
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <string>

namespace cv {
namespace v4d {
namespace import {

struct BoundingBox {
	cv::Vec3f min_;
	cv::Vec3f max_;
	cv::Vec3f center_;
	cv::Vec3f size_;
	cv::Vec3f span_;
};

class AssimpScene {
    Assimp::Importer importer_;
	const aiScene* scene_;
	BoundingBox bbox_;
	double autoScale_;
public:

AssimpScene(const std::string filename);
AssimpScene(std::vector<cv::Point3f>& vertices);
~AssimpScene();

BoundingBox boundingBox();
float autoScale();
const aiScene* scene() const;
cv::Mat_<float> verticesAsMat();
std::vector<cv::Point3f> verticesAsVector();
};

} // namespace assimp

namespace gl {

cv::Vec3f rotate3D(const cv::Vec3f& point, const cv::Vec3f& center, const cv::Vec2f& rotation);
cv::Matx44f perspective(float fov, float aspect, float zNear, float zFar);
cv::Matx44f lookAt(cv::Vec3f eye, cv::Vec3f center, cv::Vec3f up);
cv::Matx44f modelView(const cv::Vec3f& translation, const cv::Vec3f& rotationVec, const cv::Vec3f& scaleVec);

class Scene {
public:
	enum RenderMode {
		DEFAULT = 0,
		WIREFRAME = 1,
		POINTCLOUD = 2,
	};
private:
	const cv::Rect& viewport_;
	std::vector<float> gridVertices_;
	Mat volume3DData_;
	Mat skin2DData_;
	cv::v4d::import::AssimpScene* assimp_ = nullptr;
	RenderMode mode_ = DEFAULT;
	GLuint sceneFBO_ = 0;
	GLuint sceneTexture_ = 0;
	GLuint sharedDepthTexture_ = 0;
	GLuint nebulaLightingFBO_ = 0;
	GLuint nebulaLightingTexture_ = 0;
	GLuint volumeTexture_ = 0;
	GLuint skinTexture_ = 0;
	GLuint modelLightingHandles_[3] = {0, 0, 0};
	GLuint hdrHandles_[3] = {0, 0, 0};
	GLuint nebulaHandles_[3] = {0, 0, 0};
	GLuint nebulaLightingHandles_[3] = {0, 0, 0};
	cv::Vec3f lightPos_ = {1.2f, 1.0f, 2.0f};

    const string modelVertexSource_ = R"(
 	    #version 300 es
 	    layout(location = 0) in vec3 aPos;

 	    out vec3 FragPos;
		out vec2 TexCoords;
		out float Depth;
 	    uniform mat4 model;
 	    uniform mat4 view;
 	    uniform mat4 projection;

 	    void main() {
    		vec4 worldPos = model * vec4(aPos, 1.0);
			TexCoords = normalize(worldPos.xy) * 0.5 + 0.5;
    		vec4 camPos = projection * view * worldPos;
			Depth = ((camPos.z / camPos.w + 1.0) * 0.5);
			FragPos = worldPos.xyz;
		    gl_Position = projection * view * worldPos;
			gl_PointSize = 3.0;  // Set the size_ of the points
 	    }
 	)";

 	const string lightingFragmentSource_ = R"(
		#version 300 es
		
		#define RENDER_MODE_WIREFRAME 1
		#define RENDER_MODE_POINTCLOUD 2
		
		#define CONTRAST vec3(1.0, 1.0, 1.0)
		#define BRIGHTNESS vec3(0.0, 0.0, 0.0)
		#define AMBIENT_OFFSET vec3(-0.1, -0.1, -0.1)
		#define DIFFUSE_OFFSET vec3(0.1, 0.1, 0.1)
		#define SPECULAR_OFFSET vec3(0.3, 0.3, 0.3)
		
		precision highp float;
		
		in vec2 TexCoords;
		in vec3 FragPos;
 	    in float Depth;

		out vec4 FragColor;
		
		uniform vec3 viewPos;
		uniform vec3 plainColor;
		uniform mat4 invProjView;
		uniform int renderMode;
		uniform int passThrough;
		uniform sampler2D hdrBuffer;
		
		void main() {
			vec3 baseColor;
			if(length(plainColor) < 0.01) {
				baseColor = texture(hdrBuffer, TexCoords).rgb;
			} else {
				//FIXME
				baseColor = plainColor;
			}

			if(passThrough == 0) {		
				vec3 attuned;
				if (renderMode == RENDER_MODE_WIREFRAME) {
					attuned = baseColor;
				} else {
					attuned = baseColor;
				}
			
				vec3 viewDir = normalize(viewPos - FragPos);
				vec3 lightPos = viewPos - viewDir;
				vec3 lightDir = normalize(lightPos - FragPos);
				vec3 reflectDir = reflect(-lightDir, normalize(FragPos));
				float diff = max(dot(normalize(FragPos), lightDir), 0.0);
				float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
				
				vec3 ambient = attuned + AMBIENT_OFFSET;
				vec3 diffuse = (attuned + DIFFUSE_OFFSET) * diff;
				vec3 specular = (attuned + SPECULAR_OFFSET) * spec;
			
			
				float viewDist = length(viewPos - FragPos);
				vec3 finalColor = (((ambient + diffuse + specular) + BRIGHTNESS) * CONTRAST) / (1.0 + viewDist);
				
				FragColor = vec4(finalColor, 1.0);
				gl_FragDepth = Depth;
			} else {
				FragColor = vec4(baseColor, 1.0);
				gl_FragDepth = Depth;
			}
		}
	)";

 	const string textureVertexSource = R"(
		#version 300 es
		precision highp float;
		
		layout (location = 0) in vec2 aPos;
		layout (location = 1) in vec2 aTexCoords;
		
		out vec2 TexCoords;
		
		void main()
		{
			gl_Position = vec4(aPos, 0.0, 1.0);
			TexCoords = aTexCoords;
		}
	)";

    const string depthVertexSource_ = R"(
 	    #version 300 es
 	    layout(location = 0) in vec2 aPos;
		layout (location = 1) in vec2 aTexCoords;

 	    out vec3 FragPos;
		out vec2 TexCoords;
		out float Depth;

		uniform mat4 invView;
		uniform mat4 invProjection;
		uniform sampler2D depthBuffer;		

 	    void main() {
			Depth = texture(depthBuffer, aTexCoords).r;
			float z = 2.0 * 0.1 * 100.0 / (100.0 + 0.1 - (2.0 * Depth - 1.0) * (100.0 - 0.1));
    		vec4 screenPosWithZ = vec4(aPos, z, 1.0);
			vec4 camSpace = invProjection * screenPosWithZ;
			vec4 cartesian = camSpace / camSpace.w;
			vec4 worldPosition = invView * cartesian;

    		FragPos = worldPosition.xyz;
			gl_Position = vec4(aPos, 0, 1.0);
			TexCoords = aTexCoords;
 	    }
 	)";

 	const string hdrFragmentSource_ = R"(
		#version 300 es
		precision highp float;
		
		in vec2 TexCoords;
		uniform sampler2D hdrBuffer;
		uniform sampler2D depthBuffer;		
		uniform int passThrough;
		out vec4 FragColor;
		const float gamma = 2.2;

		void main()
		{
			vec3 hdrColor = texture(hdrBuffer, TexCoords).rgb;
			if(passThrough == 0) {
			float depth = texture(depthBuffer, TexCoords).r; // Sample the depth value
			
				// Check if the fragment is at the maximum depth (i.e., it's background)
				if(depth < 1.0)
				{
					hdrColor *= depth; // Modulate the color by depth
				}
			}		
			vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
			if(passThrough == 0) {
				mapped = pow(mapped, vec3(1.0 / gamma));
			}
			FragColor = vec4(mapped, 1.0);
		}
	)";

// 	const string fogFragmentSource_ = R" +
//			// Compute the fog factor
//s			float fogFactor = (fogFar - depth) / (fogFar - fogNear);
//			fogFactor = clamp(fogFactor, 0.0, 1.0);
//
//			// Mix the color with the fog color based on the fog factor
//			vec3 finalColor = mix(fogColor, color, fogFactor);
//
//			FragColor = vec4(finalColor, 1.0);
//		}
//		)";
//
 	const string volumeVertexSource_ = R"(
		#version 300 es
		precision highp float;
		layout(location = 0) in vec4 aPos;

		out vec3 TexCoord;
		void main() {
			gl_Position = vec4(aPos.xyz, 1.0);
			TexCoord = aPos.xyz * 0.5 + 0.5;
		}

	)";


 	const string nebulaFragmentSource_ = R"(
#version 300 es
precision highp float;

uniform sampler3D noiseTex;
uniform vec3 camPos;
uniform mat3 camRot;
uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 invPersMatrix;

in vec3 TexCoord;    
out vec4 FragColor;

float getNoise(vec3 pos) {
	return texture(noiseTex, pos * 0.5 + 0.5).x;
}

vec3 getColor(float density) {
    vec3 color1 = vec3(0.5, 0.0, 0.5); // purple
    vec3 color2 = vec3(1.0, 0.5, 0.0); // orange
    return mix(color1, color2, density);
}

void main() {
    vec4 ndc = vec4(
        (gl_FragCoord.x / 1920.0 - 0.5) * 2.0,
        (gl_FragCoord.y / 1080.0 - 0.5) * 2.0,
        (gl_FragCoord.z - 0.5) * 2.0,
        1.0);
		vec4 clip = invPersMatrix * ndc;
    	vec3 vertex = (clip / clip.w).xyz;

      vec3 dir = normalize(vertex - camPos);
	  float dist = length(camPos);
	  vec3 pos = camPos;
	  vec4 color = vec4(0.0);
      for(int i = 0; i < 256; i++) {
        float density = pow(getNoise(pos), 8.0);
		float r = pow(getNoise(pos.yzx), 4.0);
		float g = pow(getNoise(pos.zxy), 8.0);
		float b = pow(getNoise(pos.xzy), 2.0);
/*		float y = 0.299 * density + 0.587 * density + 0.114 * density;
		float u = 0.492 * (density-y); 
		float v = 0.877 * (density-y);*/
        color += vec4(r, g, b, density * 0.8) * 0.05;
        pos = ((camPos / 40.0) + ((dir / 40.0) * (float(i) + 1.0) * 0.1) / 2.0);
	  }

	  vec3 scaledPos = camPos / 10.0;
	  vec3 scaledDir = dir / 10.0;
	  float v = pow(getNoise((scaledPos + scaledDir) / 2.0) , 8.0);
//	  float v = getNoise(dir);
//      FragColor = vec4(v, v, v, 1.0);
//      FragColor = vec4(dir * 0.5 + 0.5, 1.0);
		FragColor = color / vec4(4.0);
		gl_FragDepth = 0.9 + pow(((v * 0.5 + 0.5) / 10.0), 8.0);
}


)";
 	void createNebulaLightingObjects();
 	void creatSkinTexture(Mat& textureData);
 	void creatVolumeTexture(Mat& textureData);
 	void createSceneObjects();
public:
	Scene(const cv::Rect& viewport);
	virtual ~Scene();
	void reset();
	bool load(const std::vector<Point3f>& points);
	bool load(const std::string& filename);
	void render(const cv::Vec3f& cameraPosition, const cv::Matx33f& cameraRotation, const cv::Matx44f& projection, const cv::Matx44f& view, const cv::Matx44f& modelView);

	float autoScale() const {
		return assimp_->autoScale();
	}

	cv::Vec3f autoCenter() const {
		return assimp_->boundingBox().center_;
	}

	RenderMode getMode() const {
		return mode_;
	}

	void setMode(RenderMode mode) {
		mode_ = mode;
	}

	cv::Vec3f lightPosition() const {
		return lightPos_;
	}

	void setLightPosition(cv::Vec3f pos) {
		lightPos_ = pos;
	}

};

} /* namespace gl */
} /* namespace v4d */
} /* namespace cv */

#endif /* MODULES_V4D_SRC_SCENE_HPP_ */
