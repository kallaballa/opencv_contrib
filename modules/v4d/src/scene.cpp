// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>


#include "../include/opencv2/v4d/scene.hpp"
#include "../third/PerlinNoise/PerlinNoise.hpp"
#include <iostream>
#include <assimp/postprocess.h>
#include <opencv2/calib3d.hpp>
#include <functional>
#include <vector>
#include <cmath>
#include <random>

namespace cv {
namespace v4d {
namespace import {

static BoundingBox calculateBoundingBox(const aiMesh* m) {
	cv::Vec3f min;
	cv::Vec3f max;
    for (unsigned int i = 0; i < m->mNumVertices; ++i) {
        aiVector3D vertex = m->mVertices[i];
        if (i == 0) {
            min = max = cv::Vec3f(vertex.x, vertex.y, vertex.z);
        } else {
            min[0] = std::min(min[0], vertex.x);
            min[1] = std::min(min[1], vertex.y);
            min[2] = std::min(min[2], vertex.z);

            max[0] = std::max(max[0], vertex.x);
            max[1] = std::max(max[1], vertex.y);
            max[2] = std::max(max[2], vertex.z);
        }
    }
    cv::Vec3f center = (min + max) / 2.0f;
    cv::Vec3f size = max - min;
    cv::Vec3f span(std::max(center[0] - min[0], max[0] - center[0]),
    		std::max(center[1] - min[1], max[1] - center[1]),
			std::max(center[2] - min[2], max[2] - center[2]));

    cerr << "min: " << min << endl;
    cerr << "max: " << max << endl;
    cerr << "center: " << center << endl;
    cerr << "size: " << size << endl;
    cerr << "span: " << span << endl;
    return {min, max, center, size, span};
}

static float calculateAutoScale(BoundingBox bbox) {
	float maxDimension = std::max(bbox.span_[0], std::max(bbox.span_[1], bbox.span_[2]));
	cerr << "scale: " << 1.0f	/maxDimension << endl;
    return 0.1f / (maxDimension * 2.0);
}

static void recurse_node(const aiNode* node, const aiScene* scene, cv::Mat_<float>& allVertices) {
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
            aiVector3D aiVertex = mesh->mVertices[j];
            cv::Mat_<float> vertex = (cv::Mat_<float>(1, 3) << aiVertex.x, aiVertex.y, aiVertex.z);
            allVertices.push_back(vertex);
        }
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        recurse_node(node->mChildren[i], scene, allVertices);
    }
}

static void recurse_node(const aiNode* node, const aiScene* scene, std::vector<cv::Point3f>& allVertices) {
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        for (unsigned int j = 0; j < mesh->mNumVertices; ++j) {
            aiVector3D aiVertex = mesh->mVertices[j];
            cv::Point3f vertex(aiVertex.x, aiVertex.y, aiVertex.z);
            allVertices.push_back(vertex);
        }
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        recurse_node(node->mChildren[i], scene, allVertices);
    }
}

AssimpScene::AssimpScene(const std::string filename) {
    scene_ = importer_.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenNormals);

    if (!scene_ || (scene_->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene_->mRootNode) {
        throw new std::runtime_error("Unable to load scene: " + filename);
    }
	bbox_ = calculateBoundingBox(scene_->mMeshes[0]);
	autoScale_= calculateAutoScale(bbox_);
}

AssimpScene::AssimpScene(std::vector<cv::Point3f>& vertices) {
    if (vertices.size() % 3 != 0) {
    	vertices.resize(vertices.size() / 3);
    }

    aiScene* scene = new aiScene();
    aiMesh* mesh = new aiMesh();

    // Set vertices
    mesh->mVertices = new aiVector3D[vertices.size()];
    for (size_t i = 0; i < vertices.size(); ++i) {
        mesh->mVertices[i] = aiVector3D(vertices[i].x, vertices[i].y, vertices[i].z);
    }
    mesh->mNumVertices = static_cast<unsigned int>(vertices.size());

    // Generate normals
    mesh->mNormals = new aiVector3D[mesh->mNumVertices];
    std::fill(mesh->mNormals, mesh->mNormals + mesh->mNumVertices, aiVector3D(0.0f, 0.0f, 0.0f));

    size_t numFaces = vertices.size() / 3;  // Assuming each face has 3 vertices
    mesh->mFaces = new aiFace[numFaces];
    mesh->mNumFaces = static_cast<unsigned int>(numFaces);

    for (size_t i = 0; i < numFaces; ++i) {
        aiFace& face = mesh->mFaces[i];
        face.mIndices = new unsigned int[3];  // Assuming each face has 3 vertices
        face.mIndices[0] = static_cast<unsigned int>(3 * i);
        face.mIndices[1] = static_cast<unsigned int>(3 * i + 1);
        face.mIndices[2] = static_cast<unsigned int>(3 * i + 2);
        face.mNumIndices = 3;

        // Calculate normal for this face
        aiVector3D edge1 = mesh->mVertices[face.mIndices[1]] - mesh->mVertices[face.mIndices[0]];
        aiVector3D edge2 = mesh->mVertices[face.mIndices[2]] - mesh->mVertices[face.mIndices[0]];
        aiVector3D normal = edge1 ^ edge2;  // Cross product
        normal.Normalize();

        // Assign the computed normal to all three vertices of the triangle
        mesh->mNormals[face.mIndices[0]] = normal;
        mesh->mNormals[face.mIndices[1]] = normal;
        mesh->mNormals[face.mIndices[2]] = normal;
    }

    // Attach the mesh to the scene
    scene->mMeshes = new aiMesh*[1];
    scene->mMeshes[0] = mesh;
    scene->mNumMeshes = 1;

    // Create a root node and attach the mesh
    scene->mRootNode = new aiNode();
    scene->mRootNode->mMeshes = new unsigned int[1]{0};
    scene->mRootNode->mNumMeshes = 1;
	bbox_ = calculateBoundingBox(mesh);
	autoScale_= calculateAutoScale(bbox_);
	scene_ = scene;
}



AssimpScene::~AssimpScene() {
    if (scene_) {
        for (unsigned int i = 0; i < scene_->mNumMeshes; ++i) {
            delete[] scene_->mMeshes[i]->mVertices;
            delete[] scene_->mMeshes[i]->mNormals;
            for (unsigned int j = 0; j < scene_->mMeshes[i]->mNumFaces; ++j) {
                delete[] scene_->mMeshes[i]->mFaces[j].mIndices;
            }
            delete[] scene_->mMeshes[i]->mFaces;
            delete scene_->mMeshes[i];
        }

        delete[] scene_->mMeshes;
        delete scene_->mRootNode;
        delete scene_;
    }
}

BoundingBox AssimpScene::boundingBox() {
	return bbox_;
}

float AssimpScene::autoScale() {
	return autoScale_;
}

const aiScene* AssimpScene::scene() const {
	return scene_;
}

cv::Mat_<float> AssimpScene::verticesAsMat() {
	cv::Mat_<float> allVertices;
	recurse_node(scene_->mRootNode, scene_, allVertices);
	return allVertices;
}

std::vector<cv::Point3f> AssimpScene::verticesAsVector() {
	std::vector<cv::Point3f> allVertices;
	import::recurse_node(scene_->mRootNode, scene_, allVertices);
	return allVertices;
}
} // namespace assimp

namespace gl {

cv::Vec3f rotate3D(const cv::Vec3f& point, const cv::Vec3f& center, const cv::Vec2f& rotation)
{
    cv::Matx33f rotationMatrix;
    cv::Rodrigues(cv::Vec3f(rotation[0], rotation[1], 0.0f), rotationMatrix);

    cv::Vec3f translatedPoint = point - center;
    cv::Vec3f rotatedPoint = rotationMatrix * translatedPoint;
    rotatedPoint += center;

    return rotatedPoint;
}

cv::Matx44f perspective(float fov, float aspect, float zNear, float zFar) {
    float tanHalfFovy = tan(fov / 2.0f);

    cv::Matx44f projection = cv::Matx44f::eye();
    projection(0, 0) = 1.0f / (aspect * tanHalfFovy);
    projection(1, 1) = 1.0f / (tanHalfFovy); // Invert the y-coordinate
    projection(2, 2) = -(zFar + zNear) / (zFar - zNear); // Invert the z-coordinate
    projection(2, 3) = -1.0f;
    projection(3, 2) = -(2.0f * zFar * zNear) / (zFar - zNear);
    projection(3, 3) = 0.0f;

    return projection;
}

cv::Matx44f lookAt(cv::Vec3f eye, cv::Vec3f center, cv::Vec3f up) {
    cv::Vec3f f = cv::normalize(center - eye);
    cv::Vec3f s = cv::normalize(f.cross(up));
    cv::Vec3f u = s.cross(f);

    cv::Matx44f view = cv::Matx44f::eye();
    view(0, 0) = s[0];
    view(0, 1) = u[0];
    view(0, 2) = -f[0];
    view(0, 3) = 0.0f;
    view(1, 0) = s[1];
    view(1, 1) = u[1];
    view(1, 2) = -f[1];
    view(1, 3) = 0.0f;
    view(2, 0) = s[2];
    view(2, 1) = u[2];
    view(2, 2) = -f[2];
    view(2, 3) = 0.0f;
    view(3, 0) = -s.dot(eye);
    view(3, 1) = -u.dot(eye);
    view(3, 2) = f.dot(eye);
    view(3, 3) = 1.0f;

    return view;
}

cv::Matx44f modelView(const cv::Vec3f& translation, const cv::Vec3f& rotationVec, const cv::Vec3f& scaleVec) {
    cv::Matx44f scaleMat(
    		scaleVec[0], 0.0, 0.0, 0.0,
            0.0, scaleVec[1], 0.0, 0.0,
            0.0, 0.0, scaleVec[2], 0.0,
            0.0, 0.0, 0.0, 1.0);

    cv::Matx44f rotXMat(
            1.0, 0.0, 0.0, 0.0,
            0.0, cos(rotationVec[0]), -sin(rotationVec[0]), 0.0,
            0.0, sin(rotationVec[0]), cos(rotationVec[0]), 0.0,
            0.0, 0.0, 0.0, 1.0);

    cv::Matx44f rotYMat(
            cos(rotationVec[1]), 0.0, sin(rotationVec[1]), 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin(rotationVec[1]), 0.0,cos(rotationVec[1]), 0.0,
            0.0, 0.0, 0.0, 1.0);

    cv::Matx44f rotZMat(
            cos(rotationVec[2]), -sin(rotationVec[2]), 0.0, 0.0,
            sin(rotationVec[2]), cos(rotationVec[2]), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0);

    cv::Matx44f translateMat(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            translation[0], translation[1], translation[2], 1.0);

    return translateMat * rotXMat * rotYMat * rotZMat * scaleMat;
}

namespace detail {
constexpr static GLuint NO_OBJECT = std::numeric_limits<GLuint>::max();
static void draw_mesh(aiMesh* mesh, Scene::RenderMode mode) {
    static GLuint VAO = NO_OBJECT;
    if(VAO == NO_OBJECT) {
    	glGenVertexArrays(1, &VAO);
    }
	glBindVertexArray(VAO);

    static GLuint VBO = NO_OBJECT;
    if(VBO == NO_OBJECT) {
    	glGenBuffers(1, &VBO);
    }

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(float), mesh->mVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	static GLuint VBO2 = NO_OBJECT;
	if(VBO2 == NO_OBJECT) {
    	glGenBuffers(1, &VBO2);
    }
	glBindBuffer(GL_ARRAY_BUFFER, VBO2);
	glBufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * 3 * sizeof(float), mesh->mNormals, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);

    if (mesh->HasFaces()) {
        std::vector<unsigned int> indices;
		for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
			aiFace face = mesh->mFaces[i];
			for (unsigned int j = 0; j < face.mNumIndices; j++)
				indices.push_back(face.mIndices[j]);
		}
        if (mode != Scene::RenderMode::DEFAULT) {
            static GLuint EBO = NO_OBJECT;
            std::vector<unsigned int> modifiedIndices;

			// Duplicate vertices for wireframe rendering or point rendering
			for (size_t i = 0; i < indices.size(); i += 3) {
				if (mode == Scene::RenderMode::WIREFRAME) {
					// Duplicate vertices for wireframe rendering
					modifiedIndices.push_back(indices[i]);
					modifiedIndices.push_back(indices[i + 1]);

					modifiedIndices.push_back(indices[i + 1]);
					modifiedIndices.push_back(indices[i + 2]);

					modifiedIndices.push_back(indices[i + 2]);
					modifiedIndices.push_back(indices[i]);
				}

				if (mode == Scene::RenderMode::POINTCLOUD) {

					// Duplicate vertices for point rendering
					modifiedIndices.push_back(indices[i]);
					modifiedIndices.push_back(indices[i + 1]);
					modifiedIndices.push_back(indices[i + 2]);
				}
			}
			if(EBO == NO_OBJECT) {
				glGenBuffers(1, &EBO);
			}

        	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        	glBufferData(GL_ELEMENT_ARRAY_BUFFER, modifiedIndices.size() * sizeof(unsigned int), &modifiedIndices[0], GL_STATIC_DRAW);

            // Draw as lines or points
            if (mode == Scene::RenderMode::WIREFRAME) {
                glDrawElements(GL_LINES, modifiedIndices.size(), GL_UNSIGNED_INT, 0);
            } else if (mode == Scene::RenderMode::POINTCLOUD) {
                glDrawElements(GL_POINTS, modifiedIndices.size(), GL_UNSIGNED_INT, 0);
            }
        } else {
            static GLuint EBO2 = NO_OBJECT;
            if(EBO2 == NO_OBJECT) {
            	glGenBuffers(1, &EBO2);
            }
        	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
        	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

            glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        }
    } else {
        glDrawArrays(GL_TRIANGLES, 0, mesh->mNumVertices);
    }
    glBindVertexArray(0);
}

static void draw_grid(std::vector<float> gridVertices) {
	GLuint gridVBO, gridVAO;

	// Generate and bind the VAO
	glGenVertexArrays(1, &gridVAO);
	glBindVertexArray(gridVAO);

	// Generate and bind the VBO
	glGenBuffers(1, &gridVBO);
	glBindBuffer(GL_ARRAY_BUFFER, gridVBO);

	// Load the grid vertices into the VBO
	glBufferData(GL_ARRAY_BUFFER, gridVertices.size() * sizeof(float), gridVertices.data(), GL_STATIC_DRAW);

	// Set the vertex attribute pointers
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// Unbind the VAO and VBO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Bind the grid VAO
	glBindVertexArray(gridVAO);

	// Draw the grid
	glDrawArrays(GL_LINES, 0, gridVertices.size() / 3);

	// Unbind the grid VAO
	glBindVertexArray(0);

	// Cleanup
    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
}

static void draw_quad()
{
    static float quadVertices[] = {
        // positions        // texture Coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };
    // setup quad VAO
    static unsigned int quadVAO = 0;
    static unsigned int quadVBO;
    if (quadVAO == 0)
    {
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

// Function to recursively draw a node and its children
static void draw_node(aiNode* node, const aiScene* scene, Scene::RenderMode mode) {
    // Draw all meshes at this node
    for(unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        draw_mesh(mesh, mode);
    }

    // Recurse for all children
    for(unsigned int i = 0; i < node->mNumChildren; i++) {
        draw_node(node->mChildren[i], scene, mode);
    }
}

// Function to draw a model
static void draw_model(const cv::v4d::import::AssimpScene& assimp, Scene::RenderMode mode) {
    // Draw the root node
    draw_node(assimp.scene()->mRootNode, assimp.scene(), mode);
}

static void make_grid_vertices(std::vector<float> gridVertices, const float gridDimension = 1.0f, const float gridStep = 0.1f) {
	const size_t numLines = (int)(2.0f * gridDimension / gridStep) + 1;
	gridVertices.resize(numLines * 12);

	for (int i = 0; i < numLines; ++i) {
		float pos = -gridDimension + i * gridStep;

		gridVertices[i*12 + 0] = -gridDimension;
		gridVertices[i*12 + 1] = 0.0f;
		gridVertices[i*12 + 2] = pos;
		gridVertices[i*12 + 3] = gridDimension;
		gridVertices[i*12 + 4] = 0.0f;
		gridVertices[i*12 + 5] = pos;

		gridVertices[i*12 + 6] = pos;
		gridVertices[i*12 + 7] = 0.0f;
		gridVertices[i*12 + 8] = -gridDimension;
		gridVertices[i*12 + 9] = pos;
		gridVertices[i*12 + 10] = 0.0f;
		gridVertices[i*12 + 11] = gridDimension;
	}
}


cv::Mat generate_3d_perlin_noise(int width, int height, int depth, const siv::PerlinNoise& noiseGenerator) {
	int sizes[3] = {height, width, depth};
	cv::Mat noiseImage = cv::Mat::zeros(3, sizes, CV_32FC1);
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float noiseValue = noiseGenerator.octave3D(float(x) /width,  float(y) / height, float(z) / depth, 16.0, 6.0);
                noiseImage.at<float>(y, x, z) = noiseValue;
            }
        }
    }
    cv::normalize(noiseImage, noiseImage, 0.0f, 1.0f, cv::NORM_MINMAX);
    return noiseImage;
}

cv::Mat generate_2d_perlin_noise(int width, int height, const siv::PerlinNoise& noiseGenerator) {
	cv::Mat noiseImage = cv::Mat::zeros(height, width, CV_32FC3);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float v = noiseGenerator.octave2D(float(x) /width,  float(y) / height, 4.0, 1.0);
			noiseImage.at<Vec3f>(y, x) = Vec3f(v, v, v);
		}
	}
    cv::normalize(noiseImage, noiseImage, 0.0f, 1.0f, cv::NORM_MINMAX);
    return noiseImage;
}


void make_3d_perlin_texture(Mat& textureData, int width, int height, int depth) {
	const siv::PerlinNoise::seed_type seed = 123456u;
	const siv::PerlinNoise generator(seed);
	textureData = generate_3d_perlin_noise(width, height, depth, generator);
}

void make_2d_perlin_texture(Mat& textureData, int width, int height) {
	const siv::PerlinNoise::seed_type seed = 123456u;
	const siv::PerlinNoise generator(seed);
	textureData = generate_2d_perlin_noise(width, height, generator);
}

/*
void initVoxels() {
    // Calculate the number of voxels needed to fill the room
    int numVoxelsX = ROOM_SIZE_X / VOXEL_SIZE;
    int numVoxelsY = ROOM_SIZE_Y / VOXEL_SIZE;
    int numVoxelsZ = ROOM_SIZE_Z / VOXEL_SIZE;

    // Resize the vector to accommodate the 3D grid voxels
    gridVoxels_.resize(numVoxelsX * numVoxelsY * numVoxelsZ * 12 * 3 * 6);  // 12 triangles per voxel, 3 vertices per triangle, 3 coordinates and 3 color values per vertex

    // Generate the grid voxels
    int index = 0;
    for (int i = 0; i < numVoxelsX; ++i) {
        for (int j = 0; j < numVoxelsY; ++j) {
            for (int k = 0; k < numVoxelsZ; ++k) {
                float posX = i * VOXEL_SIZE;
                float posY = j * VOXEL_SIZE;
                float posZ = k * VOXEL_SIZE;

                // Define the 8 corners of the voxel
                float corners[8][3] = {
                    {posX, posY, posZ},
                    {posX + VOXEL_SIZE, posY, posZ},
                    {posX + VOXEL_SIZE, posY + VOXEL_SIZE, posZ},
                    {posX, posY + VOXEL_SIZE, posZ},
                    {posX, posY, posZ + VOXEL_SIZE},
                    {posX + VOXEL_SIZE, posY, posZ + VOXEL_SIZE},
                    {posX + VOXEL_SIZE, posY + VOXEL_SIZE, posZ + VOXEL_SIZE},
                    {posX, posY + VOXEL_SIZE, posZ + VOXEL_SIZE}
                };

                // Define the color for the voxel (you can change this to whatever you want)
                float color[4] = {1.0f, 0.5f, 0.0f};  // RGB color

                // Define the 12 triangles of the voxel
                int triangles[12][3] = {
                    {0, 1, 2}, {0, 2, 3},  // bottom face
                    {4, 5, 6}, {4, 6, 7},  // top face
                    {0, 1, 5}, {0, 5, 4},  // front face
                    {2, 3, 7}, {2, 7, 6},  // back face
                    {0, 3, 7}, {0, 7, 4},  // left face
                    {1, 2, 6}, {1, 6, 5}   // right face
                };

                // Store the triangles and colors in the gridVoxels_ vector
                for (int t = 0; t < 12; ++t) {
                    for (int v = 0; v < 3; ++v) {
                        for (int c = 0; c < 3; ++c) {
                            gridVoxels_[index++] = corners[triangles[t][v]][c];
                        }
                        for (int c = 0; c < 3; ++c) {
                            gridVoxels_[index++] = color[c];
                        }
                    }
                }
            }
        }
    }

    // Create buffers/arrays
    glGenVertexArrays(1, &voxelsVAO);
    glGenBuffers(1, &voxelsVBO);

    // Bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(voxelsVAO);

    glBindBuffer(GL_ARRAY_BUFFER, voxelsVBO);
    glBufferData(GL_ARRAY_BUFFER, gridVoxels_.size() * sizeof(float), &gridVoxels_[0], GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind the VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs)
    glBindVertexArray(0);
}

void drawVoxels() {
    // Bind the VAO
    glBindVertexArray(voxelsVAO);

    // Draw the voxels
    glDrawArrays(GL_TRIANGLES, 0, gridVoxels_.size() / 3);

    // Unbind the VAO
    glBindVertexArray(0);
}
*/
}

void Scene::creatSkinTexture(Mat& textureData) {
	glGenTextures(1, &skinTexture_);
	glBindTexture(GL_TEXTURE_2D, skinTexture_);
	GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, textureData.cols, textureData.rows, 0, GL_RGB, GL_FLOAT, textureData.data));
}

void Scene::creatVolumeTexture(Mat& textureData) {
	glGenTextures(1, &volumeTexture_);
	glBindTexture(GL_TEXTURE_3D, volumeTexture_);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32F, textureData.size[0], textureData.size[1], textureData.size[2]);
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, textureData.size[0], textureData.size[1], textureData.size[2], GL_RED, GL_FLOAT, textureData.data);
}

void Scene::createSceneObjects() {
	glGenFramebuffers(1, &sceneFBO_);
	glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO_);
	glGenTextures(1, &sceneTexture_);
	glBindTexture(GL_TEXTURE_2D, sceneTexture_);
    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, viewport_.width, viewport_.height, 0, GL_RGBA, GL_FLOAT, NULL));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneTexture_, 0));
    glGenTextures(1, &sharedDepthTexture_);
    glBindTexture(GL_TEXTURE_2D, sharedDepthTexture_);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate storage for the texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, viewport_.width, viewport_.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    // Attach the texture to your framebuffer
    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sharedDepthTexture_, 0));
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

void Scene::createNebulaLightingObjects() {
	glGenFramebuffers(1, &nebulaLightingFBO_);
	glBindFramebuffer(GL_FRAMEBUFFER, nebulaLightingFBO_);
	glGenTextures(1, &nebulaLightingTexture_);
	glBindTexture(GL_TEXTURE_2D, nebulaLightingTexture_);
    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, viewport_.width, viewport_.height, 0, GL_RGBA, GL_FLOAT, NULL));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, nebulaLightingTexture_, 0));
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

Scene::Scene(const cv::Rect& viewport) : viewport_(viewport) {
	detail::make_grid_vertices(gridVertices_);
	detail::make_3d_perlin_texture(volume3DData_, viewport.width / 8.0, viewport.height / 8.0, viewport.height / 8.0);
	detail::make_2d_perlin_texture(skin2DData_, viewport.width / 16.0, viewport.height / 16.0);
}

Scene::~Scene() {
}


void Scene::reset() {
	if(modelLightingHandles_[0] > 0)
		glDeleteProgram(modelLightingHandles_[0]);
	if(modelLightingHandles_[1] > 0)
		glDeleteShader(modelLightingHandles_[1]);
	if(modelLightingHandles_[2] > 0)
		glDeleteShader(modelLightingHandles_[2]);
	if(assimp_)
		delete assimp_;
	assimp_ = nullptr;
}

bool Scene::load(const std::vector<Point3f>& points) {
	reset();
	createSceneObjects();
	creatSkinTexture(skin2DData_);
	creatVolumeTexture(volume3DData_);
	createNebulaLightingObjects();
	std::vector<Point3f> copy = points;
    assimp_ = new cv::v4d::import::AssimpScene(copy);
    cv::v4d::init_shaders(modelLightingHandles_, modelVertexSource_.c_str(), lightingFragmentSource_.c_str(), "FragColor");
    cv::v4d::init_shaders(nebulaHandles_, volumeVertexSource_.c_str(), nebulaFragmentSource_.c_str(), "FragColor");
    cv::v4d::init_shaders(nebulaLightingHandles_, depthVertexSource_.c_str(), lightingFragmentSource_.c_str(), "FragColor");
    cv::v4d::init_shaders(hdrHandles_, textureVertexSource.c_str(), hdrFragmentSource_.c_str(), "FragColor");
    return true;
}


bool Scene::load(const std::string& filename) {
	reset();
	creatSkinTexture(skin2DData_);
	creatVolumeTexture(volume3DData_);
	createSceneObjects();
	createNebulaLightingObjects();
    assimp_ = new cv::v4d::import::AssimpScene(filename);
    cv::v4d::init_shaders(modelLightingHandles_, modelVertexSource_.c_str(), lightingFragmentSource_.c_str(), "FragColor");
    cv::v4d::init_shaders(nebulaHandles_, volumeVertexSource_.c_str(), nebulaFragmentSource_.c_str(), "FragColor");
    cv::v4d::init_shaders(nebulaLightingHandles_, depthVertexSource_.c_str(), lightingFragmentSource_.c_str(), "FragColor");
    cv::v4d::init_shaders(hdrHandles_, textureVertexSource.c_str(), hdrFragmentSource_.c_str(), "FragColor");

    return true;
}

void Scene::render(const cv::Vec3f& cameraPosition, const cv::Vec3f& cameraDirection, const cv::Matx33f& cameraRotation, const cv::Matx44f& projection, const cv::Matx44f& view, const cv::Matx44f& modelView) {
	cerr << cameraRotation << endl;
	cerr << cameraPosition << endl;
	GL_CHECK(glViewport(viewport_.x, viewport_.y, viewport_.width, viewport_.height));
	GL_CHECK(glDepthMask(GL_TRUE));
	GL_CHECK(glEnable(GL_DEPTH_TEST));
	GL_CHECK(glEnable(GL_BLEND));
	GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA))
    GL_CHECK(glEnable(GL_VERTEX_PROGRAM_POINT_SIZE));
	GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO_));
	GL_CHECK(glActiveTexture(GL_TEXTURE0));
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, sceneTexture_));
	GL_CHECK(glActiveTexture(GL_TEXTURE1));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, sharedDepthTexture_));
    GL_CHECK(glBindVertexArray(0));
    GL_CHECK(glActiveTexture(GL_TEXTURE2));
    GL_CHECK(glBindTexture(GL_TEXTURE_3D, volumeTexture_));
    GL_CHECK(glActiveTexture(GL_TEXTURE3));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, skinTexture_));
    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneTexture_, 0));
	GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sharedDepthTexture_, 0));
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
	GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glUseProgram(modelLightingHandles_[0]));
	GL_CHECK(glUniformMatrix4fv(glGetUniformLocation(modelLightingHandles_[0], "projection"), 1, GL_FALSE, projection.val));
	GL_CHECK(glUniformMatrix4fv(glGetUniformLocation(modelLightingHandles_[0], "view"), 1, GL_FALSE, view.val));
	GL_CHECK(glUniformMatrix4fv(glGetUniformLocation(modelLightingHandles_[0], "model"), 1, GL_FALSE, modelView.val));
	GL_CHECK(glUniform3fv(glGetUniformLocation(modelLightingHandles_[0], "viewPos"), 1, cameraPosition.val));
	GL_CHECK(glUniform1i(glGetUniformLocation(modelLightingHandles_[0], "renderMode"), mode_));
	GL_CHECK(glUniform1i(glGetUniformLocation(modelLightingHandles_[0], "passThrough"), 0));
//	cv::Vec3f baseColor(0.3, 0.3, 1.0);
//	GL_CHECK(glUniform3fv(glGetUniformLocation(modelLightingHandles_[0], "plainColor"), 1, baseColor.val));
	GL_CHECK(glUniform1i(glGetUniformLocation(modelLightingHandles_[0], "hdrBuffer"), 3));
	GL_CHECK(glUniformMatrix4fv(glGetUniformLocation(modelLightingHandles_[0], "invProjView"), 1, GL_FALSE, (view * projection).inv().val));

	detail::draw_model(*assimp_, mode_);
	detail::draw_grid(gridVertices_);

	GL_CHECK(glUseProgram(nebulaHandles_[0]));
    GL_CHECK(glUniform1i(glGetUniformLocation(nebulaHandles_[0], "noise3DBuffer"), 2));
    GL_CHECK(glUniform3fv(glGetUniformLocation(nebulaHandles_[0], "viewPos"), 1, cameraPosition.val));
	GL_CHECK(glUniformMatrix4fv(glGetUniformLocation(nebulaHandles_[0], "invProjView"), 1, GL_FALSE, (view * projection).inv().val));
	detail::draw_quad();

//	GL_CHECK(glDepthMask (GL_FALSE));
	GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, nebulaLightingFBO_));
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, nebulaLightingTexture_));
    GL_CHECK(glActiveTexture(GL_TEXTURE1));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, sceneTexture_));
    GL_CHECK(glActiveTexture(GL_TEXTURE2));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, sharedDepthTexture_));

	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	GL_CHECK(glUseProgram(nebulaLightingHandles_[0]));
	GL_CHECK(glUniform1i(glGetUniformLocation(nebulaLightingHandles_[0], "hdrBuffer"), 1));
	GL_CHECK(glUniform1i(glGetUniformLocation(nebulaLightingHandles_[0], "depthBuffer"), 2));
	GL_CHECK(glUniformMatrix4fv(glGetUniformLocation(nebulaLightingHandles_[0], "invProjection"), 1, GL_FALSE, projection.inv().val));
	GL_CHECK(glUniform3fv(glGetUniformLocation(nebulaLightingHandles_[0], "viewPos"), 1, cameraPosition.val));
	GL_CHECK(glUniform3fv(glGetUniformLocation(nebulaLightingHandles_[0], "viewDir"), 1, cameraDirection.val));
	GL_CHECK(glUniform1i(glGetUniformLocation(nebulaLightingHandles_[0], "renderMode"), mode_));
	GL_CHECK(glUniform1i(glGetUniformLocation(nebulaLightingHandles_[0], "passThrough"), 0));
	GL_CHECK(glUniform3fv(glGetUniformLocation(nebulaLightingHandles_[0], "plainColor"), 1, cv::Vec3f(0.0, 0.0, 0.0).val));
	detail::draw_quad();

	GL_CHECK(glDepthMask (GL_FALSE));
	GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, nebulaLightingTexture_));
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	GL_CHECK(glUseProgram(hdrHandles_[0]));
	GL_CHECK(glUniform1i(glGetUniformLocation(hdrHandles_[0], "hdrBuffer"), 0));
	GL_CHECK(glUniform1i(glGetUniformLocation(hdrHandles_[0], "passThrough"), 0));
	detail::draw_quad();
}

} /* namespace gl */
} /* namespace v4d */
} /* namespace cv */
