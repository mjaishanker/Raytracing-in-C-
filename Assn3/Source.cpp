//
//		          Programming Assignment #2 
//
//			        Victor Zordan
//		
//		
//
/***************************************************************************/

												   /* Include needed files */
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <limits>
#include <cassert>
#include <vector>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <GL/glut.h>
using namespace std;

#include <GL/glut.h>   // The GL Utility Toolkit (Glut) Header

#define WIDTH 500
#define HEIGHT 500

// Scale the fit the model in window
float scale = sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT);

int x_last, y_last, D = -400;
int translateSwitch = 0, scaleSwitch = 0, rotSwitch = 0;
float centerX, centerY, centerZ, centerW, maxValue = 0.0, minValue = 0.0;
bool viewSwitch = false;
bool render_go = false;
bool BumpMapON = false;
bool PhongSmoothON = false;
// Perspective view matrix
float presMat[16]
= {
	1.0, 0, 0, 0,
	0, 1.0, 0, 0,
	0, 0, 1.0, 0,
	0, 0, 1.0 / D, 0
};

// Vertices structure to hold main vertices of the model
struct verticesStruct {
	// w is 1 if orthogonal view and z/d if perspective vew
	float x, y, z, w;
	verticesStruct() {};
	verticesStruct(float X, float Y, float Z) { x = X; y = Y; z = Z; };
};

// Face struct to hold the indices of the faces vertices
struct faceStruct {
	int v1, v2, v3;
	verticesStruct Ambient;
	verticesStruct Diffuse;
	verticesStruct Specular;
};

// Structure for holding texture vertices
struct vertTextStruct {
	float x, y;
};

//Structure to hold the normal vertices
struct normalStruct {
	float x, y, z;
};

struct Ray {
	verticesStruct Origin;
	verticesStruct Direction;
};

struct Light {
	verticesStruct Position;
	verticesStruct Ambient;
	verticesStruct Diffuse;
	verticesStruct Specular;
};

verticesStruct Eye = { 0, 0 ,0 };


// Vectors to hold all the vertices and indices
std::vector< verticesStruct > vertices;
std::vector< verticesStruct > MainVertices;
std::vector< vertTextStruct > uvs;
std::vector< normalStruct > normals;
std::vector< faceStruct > faces;

verticesStruct colorArray[WIDTH][HEIGHT];
vector<Light> lights;
// function to center the model when trasnformation is done
void centerVertices() {
	float xSum = 0.0, ySum = 0.0, zSum = 0.0, wSum = 0.0;
	int vertexSize = MainVertices.size();

	for (size_t i = 0; i < MainVertices.size(); ++i) {
		xSum += MainVertices[i].x;
		ySum += MainVertices[i].y;
		zSum += MainVertices[i].z;
		wSum += MainVertices[i].w;
	}
	centerX = xSum / vertexSize;
	centerY = ySum / vertexSize;
	centerZ = zSum / vertexSize;
	centerW = wSum / vertexSize;

	for (size_t i = 0; i < MainVertices.size(); i++) {
		MainVertices[i].x = MainVertices[i].x - centerX;
		MainVertices[i].y = MainVertices[i].y - centerY;
		//MainVertices[i].z = MainVertices[i].z - centerZ - maxValue - minValue;
		MainVertices[i].z = MainVertices[i].z - centerZ - maxValue - minValue;
		MainVertices[i].w = MainVertices[i].w - centerW;
	}
}

// Function to center and normalize the vertices at the begining of the display
void normalizeVertices() {
	float xSum = 0.0, ySum = 0.0, zSum = 0.0, wSum = 0.0;
	float centerX, centerY, centerZ, centerW;
	int vertexSize = MainVertices.size();

	vector<verticesStruct> returnVert;

	for (size_t i = 0; i < MainVertices.size(); ++i) {
		xSum += MainVertices[i].x;
		ySum += MainVertices[i].y;
		zSum += MainVertices[i].z;
		wSum += MainVertices[i].w;
	}
	centerX = xSum / vertexSize;
	centerY = ySum / vertexSize;
	centerZ = zSum / vertexSize;
	centerW = wSum / vertexSize;

	for (size_t i = 0; i < MainVertices.size(); i++) {
		maxValue = max(maxValue, abs(MainVertices[i].x));
		maxValue = max(maxValue, abs(MainVertices[i].y));
		maxValue = max(maxValue, abs(MainVertices[i].z));
		minValue = min(minValue, abs(MainVertices[i].x));
		minValue = min(minValue, abs(MainVertices[i].y));
		minValue = min(minValue, abs(MainVertices[i].z));
	}

	for (size_t i = 0; i < MainVertices.size(); i++) {
		MainVertices[i].x = MainVertices[i].x - centerX;
		MainVertices[i].y = MainVertices[i].y - centerY;
		MainVertices[i].z = MainVertices[i].z - centerZ - maxValue;
		MainVertices[i].w = MainVertices[i].w - centerW;
	}
	// Scaling the model
	scale = scale / (maxValue + minValue); // 2*maxValue


	for (size_t i = 0; i < MainVertices.size(); i++) {
		MainVertices[i].x = MainVertices[i].x * scale;
		MainVertices[i].y = MainVertices[i].y * scale;
		MainVertices[i].z = MainVertices[i].z * scale;
		printf("%f, %f, %f\n", MainVertices[i].x, MainVertices[i].y, MainVertices[i].z);
	}
}
double dotProduct(verticesStruct v0, verticesStruct v1) {
	double DP = (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z);
	return DP;
}

verticesStruct crossProduct(verticesStruct v0, verticesStruct v1) {
	verticesStruct CP;
	CP.x = v0.y * v1.z - v0.z * v1.y;
	CP.y = v0.z * v1.x - v0.x * v1.z;
	CP.z = v0.x * v1.y - v0.y * v1.x;
	return CP;
}

verticesStruct vectMult(verticesStruct v0, float t) {
	verticesStruct CP;
	CP.x = v0.x * t;
	CP.y = v0.y * t;
	CP.z = v0.z * t;
	return CP;
}

normalStruct floatNormalMult(normalStruct v0, float t) {
	normalStruct CP;
	CP.x = v0.x * t;
	CP.y = v0.y * t;
	CP.z = v0.z * t;
	return CP;
}

verticesStruct ColorMultiplication(verticesStruct v0, verticesStruct v1) {
	verticesStruct CP;
	CP.x = v0.x * v0.x;
	CP.y = v0.y * v0.y;
	CP.z = v0.z * v0.z;
	return CP;
}

verticesStruct vectorScalarAdd(verticesStruct v0, float t) {
	verticesStruct CP;
	CP.x = v0.x + t;
	CP.y = v0.y + t;
	CP.z = v0.z + t;
	return CP;
}

float vectMag(verticesStruct v0) {
	verticesStruct CP;
	CP.x = v0.x * v0.x;
	CP.y = v0.y * v0.y;
	CP.z = v0.z * v0.z;
	float t = CP.x + CP.y + CP.z;
	t = sqrt(t);
	return t;
}

verticesStruct vectSub(verticesStruct v0, verticesStruct v1) {
	verticesStruct CP;
	CP.x = v0.x - v1.x;
	CP.y = v0.y - v1.y;
	CP.z = v0.z - v1.z;
	return CP;
}

verticesStruct vectAdd(verticesStruct v0, verticesStruct v1) {
	verticesStruct CP;
	CP.x = v0.x + v1.x;
	CP.y = v0.y + v1.y;
	CP.z = v0.z + v1.z;
	return CP;
}

verticesStruct vectNormalize(verticesStruct v0) {
	float vectorLength = vectMag(v0);
	verticesStruct CP;
	CP.x = v0.x / vectorLength;
	CP.y = v0.y / vectorLength;
	CP.z = v0.z / vectorLength;
	return CP;
}
// Parser to read the object file to take vertices.
void readObjFile(const char* path, std::vector < verticesStruct >& main_vertices,
	std::vector < vertTextStruct >& main_uvs, std::vector < normalStruct >& main_normals) {

	std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
	std::vector<unsigned int> smoothGroup;
	std::vector<verticesStruct> temp_vertices;
	std::vector<vertTextStruct> temp_textures;
	std::vector<normalStruct> temp_normals;

	fstream file;
	file.open(path);
	string line;
	if (!file) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}
	fstream fin(path, fstream::in);
	while (std::getline(file, line)) {
		std::istringstream lineSS(line);
		std::string lineType;
		lineSS >> lineType;

		if (lineType == "v") {
			verticesStruct vertex;
			lineSS >> vertex.x >> vertex.y >> vertex.z;
			temp_vertices.push_back(vertex);
		}

		// texture
		if (lineType == "vt")
		{
			vertTextStruct uv;
			lineSS >> uv.x >> uv.y;
			temp_textures.push_back(uv);
		}

		// normal
		if (lineType == "vn")
		{
			normalStruct normal;
			lineSS >> normal.x >> normal.y >> normal.z;
			temp_normals.push_back(normal);
		}

		// face
		if (lineType == "f")
		{
			std::string tempStr;
			while (lineSS >> tempStr)
			{
				std::istringstream ref(tempStr);
				std::string vLine, vtLine, vnLine;
				std::getline(ref, vLine, '/');
				std::getline(ref, vtLine, '/');
				std::getline(ref, vnLine, '/');
				int v = atoi(vLine.c_str());
				int vt = atoi(vtLine.c_str());
				int vn = atoi(vnLine.c_str());
				vertexIndices.push_back(v);
				normalIndices.push_back(vn);
			}
		}
	}
	MainVertices = temp_vertices;
	normals = temp_normals;
	faceStruct tempFace;

	// Store the vertices indices in the vector of faces structure
	for (int i = 0; i < vertexIndices.size() - 2; i = i + 3) {
		tempFace.v1 = vertexIndices[i];
		tempFace.v2 = vertexIndices[i + 1];
		tempFace.v3 = vertexIndices[i + 2];
		faces.push_back(tempFace);
	}
}

verticesStruct calculateNormal(faceStruct f) {
	verticesStruct vector1 = vectSub(MainVertices[f.v2 - 1], MainVertices[f.v1 - 1]);
	verticesStruct vector2 = vectSub(MainVertices[f.v3 - 1], MainVertices[f.v1 - 1]);

	verticesStruct normal = crossProduct(vector1, vector2);
	normal = vectNormalize(normal);
	normal = vectNormalize(normal);
	return normal;
}

int DetermineBarycentric(verticesStruct triangleVertex, faceStruct f)
{
	float bary1, bary2, bary3;

	verticesStruct pos1, pos2, pos3;
	verticesStruct vertex1, vertex2, vertextT;

	float dot11, dot12, dot22, dot;

	pos1 = MainVertices[f.v1 - 1];
	pos2 = MainVertices[f.v2 - 1];
	pos3 = MainVertices[f.v3 - 1];

	vertex1 = vectSub(pos1, pos2);
	vertex2 = vectSub(pos3, pos2);
	vertextT = vectSub(triangleVertex, pos2);

	dot11 = dotProduct(vertex1, vertex1);
	dot12 = dotProduct(vertex1, vertex2);
	dot22 = dotProduct(vertex2, vertex2);

	dot = dot11 * dot22 - dot12 * dot12;

	float dt1, dt2;
	dt1 = dotProduct(vertextT, vertex1);
	dt2 = dotProduct(vertextT, vertex2);

	bary1 = (dot22 * dt1 - dot12 * dt2) / dot;
	bary2 = (dot11 * dt2 - dot12 * dt1) / dot;
	bary3 = 1 - bary1 - bary2;

	if (bary1 > 0.9999 && bary2 < 0.0001 && bary3 < 0.0001)
		return 0;
	if (bary2 > 0.9999 && bary1 < 0.0001 && bary3 < 0.0001)
		return 1;
	if (bary3 > 0.9999 && bary1 < 0.0001 && bary2 < 0.0001)
		return 2;
}

verticesStruct normalToVertices(normalStruct n) {
	verticesStruct v;
	v.x = n.x;
	v.y = n.y;
	v.z = n.z;
	return v;
}

verticesStruct BaryCenterCoor(verticesStruct IntersectionPoint, faceStruct f)
{
	float localBaryCoor[3];
	float dott1, dott2;
	int bary1, bary2, bary3;

	verticesStruct pos1, pos2, pos3;
	verticesStruct vertex1, vertex2, vertextT, normToVert1, normToVert2, normToVert3;

	float dot11, dot12, dot22, dot;

	pos1 = MainVertices[f.v1 - 1];
	pos2 = MainVertices[f.v2 - 1];
	pos3 = MainVertices[f.v3 - 1];

	vertex1 = vectSub(pos1, pos2);
	vertex2 = vectSub(pos3, pos2);
	vertextT = vectSub(IntersectionPoint, pos2);

	dot11 = dotProduct(vertex1, vertex1);
	dot12 = dotProduct(vertex1, vertex2);
	dot22 = dotProduct(vertex2, vertex2);

	dot = dot11 * dot22 - dot12 * dot12;

	dott1 = dotProduct(vertextT, vertex1);
	dott2 = dotProduct(vertextT, vertex2);

	localBaryCoor[0] = (dot22 * dott1 - dot12 * dott2) / dot;
	localBaryCoor[1] = (dot11 * dott2 - dot12 * dott1) / dot;
	localBaryCoor[2] = 1 - localBaryCoor[0] - localBaryCoor[1];

	bary1 = DetermineBarycentric(pos1, f);
	bary2 = DetermineBarycentric(pos2, f);
	bary3 = DetermineBarycentric(pos3, f);

	normToVert1 = vectMult(normalToVertices(normals[f.v1 - 1]), localBaryCoor[bary1]);
	normToVert2 = vectMult(normalToVertices(normals[f.v2 - 1]), localBaryCoor[bary2]);
	normToVert3 = vectMult(normalToVertices(normals[f.v3 - 1]), localBaryCoor[bary3]);
	return vectAdd(normToVert1, vectAdd(normToVert2, normToVert3));
}

struct BumpFace {
	verticesStruct v1, v2, v3;
};

BumpFace BumpMap(faceStruct f, float s) {
	verticesStruct v1 = MainVertices[f.v1 - 1];
	verticesStruct v2 = MainVertices[f.v2 - 1];
	verticesStruct v3 = MainVertices[f.v3 - 1];

	float centerX = (v1.x + v2.x + v3.x) / 3;
	float centerY = (v1.y + v2.y + v3.y) / 3;
	float centerZ = (v1.z + v2.z + v3.z) / 3;

	v1.x = v1.x - centerX;
	v1.y = v1.y - centerY;
	v1.z = v1.z - centerZ;
	v2.x = v2.x - centerX;
	v2.y = v2.y - centerY;
	v2.z = v2.z - centerZ;
	v3.x = v3.x - centerX;
	v3.y = v3.y - centerY;
	v3.z = v3.z - centerZ;

	v1 = vectMult(v1, s);
	v2 = vectMult(v2, s);
	v3 = vectMult(v3, s);

	v1.x = v1.x + centerX;
	v1.y = v1.y + centerY;
	v1.z = v1.z + centerZ;
	v2.x = v2.x + centerX;
	v2.y = v2.y + centerY;
	v2.z = v2.z + centerZ;
	v3.x = v3.x + centerX;
	v3.y = v3.y + centerY;
	v3.z = v3.z + centerZ;

	BumpFace bF;
	bF.v1 = v1;
	bF.v2 = v2;
	bF.v3 = v3;

	return bF;
}

bool insideface(verticesStruct intersection_point, BumpFace bF)
{
	verticesStruct vector1 = vectSub(bF.v2 , bF.v1);
	verticesStruct vector2 = vectSub(bF.v3 , bF.v2);

	verticesStruct normal = crossProduct(vector1, vector2);
	normal = vectNormalize(normal);

	verticesStruct  test, edge, point_edge;
	edge = vectSub(bF.v2, bF.v1);
	point_edge = vectSub(intersection_point,bF.v2);
	test = crossProduct(edge, point_edge);
	if (dotProduct(normal, test) < 0)
		return false;

	edge = vectSub(bF.v3, bF.v2);
	point_edge = vectSub(intersection_point, bF.v3);
	test = crossProduct(edge, point_edge);
	if (dotProduct(normal, test) < 0)
		return false;

	edge = vectSub(bF.v1, bF.v3);
	point_edge = vectSub(intersection_point, bF.v1);
	test = crossProduct(edge, point_edge);
	if (dotProduct(normal, test) < 0)
		return false;

	return true;
}

float intersection(Ray ray, faceStruct f)
{
	// check for two edges
	verticesStruct edge1 = vectSub(MainVertices[f.v2 - 1], MainVertices[f.v1 - 1]);
	verticesStruct edge2 = vectSub(MainVertices[f.v3 - 1], MainVertices[f.v1 - 1]);

	verticesStruct normal = crossProduct(edge1, edge2);
	normal = vectNormalize(normal);
	float NRD = dotProduct(normal, ray.Direction);

	if (fabs(NRD) < 0.001)
	{
		return NULL;
	}

	if (NRD > 0)
	{
		normal.x = -normal.x;
		normal.y = -normal.y;
		normal.z = -normal.z;


		NRD = -NRD; //dotProduct(normal, ray.Direction);
	}

	float d = dotProduct(normal, MainVertices[f.v2 - 1]);

	float t = (dotProduct(normal, ray.Origin) + d) / NRD;
	if (t < 0)
	{
		//printf("breaking here 3");
		return NULL;
	}
	verticesStruct intersection_point = vectAdd(ray.Origin, vectMult(ray.Direction, t));
	//printf("%f,%f,%f \n", intersection_point.x, intersection_point.y, intersection_point.z);
	verticesStruct  test, edge, point_edge;
	edge = vectSub(MainVertices[f.v2 - 1], MainVertices[f.v1 - 1]);
	point_edge = vectSub(intersection_point, MainVertices[f.v2 - 1]);
	test = crossProduct(edge, point_edge);
	if (dotProduct(normal, test) < 0)
		return NULL;

	edge = vectSub(MainVertices[f.v3 - 1], MainVertices[f.v2 - 1]);
	point_edge = vectSub(intersection_point, MainVertices[f.v3 - 1]);
	test = crossProduct(edge, point_edge);
	if (dotProduct(normal, test) < 0)
		return NULL;

	edge = vectSub(MainVertices[f.v1 - 1], MainVertices[f.v3 - 1]);
	point_edge = vectSub(intersection_point, MainVertices[f.v3 - 1]);
	test = crossProduct(edge, point_edge);
	if (dotProduct(normal, test) < 0)
		return NULL;

	return t;

}

//Raytrace function that does the raytracing of the obj file
void rayTrace() {
		for (int i = 0; i < HEIGHT; i++) {
			for (int j = 0; j < WIDTH; j++) {

				// Going through each pixel to get which face gets hit with e ray and assigning t
				float x = 2 * ((j + 0.5) / WIDTH) - 1;
				float y = 1 - 2 * ((i + 0.5) / HEIGHT);
				verticesStruct rayDir = { x, y, -1};
				rayDir = vectNormalize(rayDir);
				Ray ray = { Eye, rayDir };
				float min_t = 999999;
				int faceId = -1;
				for (size_t k = 0; k < faces.size(); k++) {
					float t = intersection(ray, faces[k]);
					if (t == NULL) continue;
					if (t < min_t) {

						min_t = t;
						faceId = k;
					}
				}

				// Calculate Shadow ray
				Ray Shadow;
				if (min_t != 999999) {
					verticesStruct intersection_point = vectAdd(ray.Origin, vectMult(ray.Direction, min_t));
					verticesStruct illumination = { 0, 0, 0 };
					// Go through each light
					for (int l = 0; l < lights.size(); l++) {
						verticesStruct lightDir = vectSub(lights[l].Position, intersection_point);
						// Assign object color
						verticesStruct KD = faces[faceId].Diffuse;
						verticesStruct KS = faces[faceId].Specular;
						verticesStruct KA = faces[faceId].Ambient;
						// Diffuse
						KD.x = 0.6;
						KD.y = 0.3;
						KD.z = 0.5;
						// Ambient
						KA.x = 0.2;
						KA.y = 0.05;
						KA.z = 0;
						// Specular
						KS.x = 0.99;
						KS.y = 0.99;
						KS.z = 0.99;
						lightDir = vectNormalize(lightDir); // just the direction towards the light
						Ray ShadowRay;
						ShadowRay.Direction = lightDir;
						ShadowRay.Origin = vectorScalarAdd(intersection_point, 0.001);
						// Initially set the shadow is not in
						bool inShadow = false;
						for (size_t k = 0; k < faces.size(); k++) {
							if (faceId == k)
								continue;
							else {
								float shadowT;
								// Check if there is an intersection between the face and shadow ray then its in shadow
								shadowT = intersection(ShadowRay, faces[k]);
								if (shadowT != NULL) {
									inShadow = true;
									break;
								}
							}
						}
						// If in shadow then just calc ambient
						if (inShadow == true) {
							//illumination = vectAdd(illumination, lights[l].Ambient);
							illumination.x = illumination.x + KA.x * lights[l].Ambient.x;
							illumination.y = illumination.y + KA.y * lights[l].Ambient.y;
							illumination.z = illumination.z + KA.z * lights[l].Ambient.z;
						}
						// calculate for all lights
						else {
							verticesStruct norm = calculateNormal(faces[faceId]);
							// if phong shadin is true
							if (PhongSmoothON == true) {
								// Calculate barycentric cool for phong shading
								norm = BaryCenterCoor(intersection_point, faces[faceId]);
								vectNormalize(norm);
								verticesStruct reflectiveRay = vectSub(vectMult(norm, dotProduct(norm, lightDir) * 2), lightDir);
								float diffNormLight;
								reflectiveRay = vectNormalize(reflectiveRay);
								//norm = vectNormalize(norm);
								//illumination = vectAdd(illumination, vectMult(ColorMultiplication(KD, lights[l].Diffuse), dotProduct(norm, lightDir))); // need to add object material k
								// Illumination add with diffuse
								illumination.x = illumination.x + KD.x * lights[l].Diffuse.x * dotProduct(norm, lightDir);
								illumination.y = illumination.y + KD.y * lights[l].Diffuse.y * dotProduct(norm, lightDir);
								illumination.z = illumination.z + KD.z * lights[l].Diffuse.z * dotProduct(norm, lightDir);

								// Specular
								illumination.x = illumination.x + KS.x * lights[l].Specular.x * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);
								illumination.y = illumination.y + KS.y * lights[l].Specular.y * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);
								illumination.z = illumination.z + KS.z * lights[l].Specular.z * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);

								// Ambient
								illumination.x = illumination.x + KA.x * lights[l].Ambient.x;
								illumination.y = illumination.y + KA.y * lights[l].Ambient.y;
								illumination.z = illumination.z + KA.z * lights[l].Ambient.z;
							}
							// if bump map is toggles
							if (BumpMapON == true) {
								BumpFace bF;
								bool isInside = false;
								bF = BumpMap(faces[faceId], 0.8);
								verticesStruct k;
								isInside = insideface(intersection_point, bF);
								if (isInside) {

									norm.x = norm.x * 0.1;
									norm.y = norm.y * 0.1;
									norm.z = norm.z * 0.1;

									BumpFace bF;
									bool isInside = false;
									bF = BumpMap(faces[faceId], 0.6);
									isInside = insideface(intersection_point, bF);
									//verticesStruct norm = calculateNormal(faces[faceId]);

									if (isInside)
									{
										if (PhongSmoothON == true)
											norm = BaryCenterCoor(intersection_point, faces[faceId]);
										else
											norm = calculateNormal(faces[faceId]);
									}
								}
								// Adding illumination
								verticesStruct reflectiveRay = vectSub(vectMult(norm, dotProduct(norm, lightDir) * 2), lightDir);
								float diffNormLight;
								reflectiveRay = vectNormalize(reflectiveRay);
								illumination.x = illumination.x + KD.x * lights[l].Diffuse.x * dotProduct(norm, lightDir);
								illumination.y = illumination.y + KD.y * lights[l].Diffuse.y * dotProduct(norm, lightDir);
								illumination.z = illumination.z + KD.z * lights[l].Diffuse.z * dotProduct(norm, lightDir);
								illumination.x = illumination.x + KS.x * lights[l].Specular.x * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);
								illumination.y = illumination.y + KS.y * lights[l].Specular.y * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);
								illumination.z = illumination.z + KS.z * lights[l].Specular.z * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);
								illumination.x = illumination.x + KA.x * lights[l].Ambient.x;
								illumination.y = illumination.y + KA.y * lights[l].Ambient.y;
								illumination.z = illumination.z + KA.z * lights[l].Ambient.z;
							}
							else {
								// Adding illumination without bump mapping
								verticesStruct norm = calculateNormal(faces[faceId]);
								norm = vectNormalize(norm);
								verticesStruct reflectiveRay = vectSub(vectMult(norm, dotProduct(norm, lightDir) * 2), lightDir);
								float diffNormLight;
								reflectiveRay = vectNormalize(reflectiveRay);
								//norm = vectNormalize(norm);
								//illumination = vectAdd(illumination, vectMult(ColorMultiplication(KD, lights[l].Diffuse), dotProduct(norm, lightDir))); // need to add object material k
								illumination.x = illumination.x + KD.x * lights[l].Diffuse.x * dotProduct(norm, lightDir);
								illumination.y = illumination.y + KD.y * lights[l].Diffuse.y * dotProduct(norm, lightDir);
								illumination.z = illumination.z + KD.z * lights[l].Diffuse.z * dotProduct(norm, lightDir);

								illumination.x = illumination.x + KS.x * lights[l].Specular.x * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);
								illumination.y = illumination.y + KS.y * lights[l].Specular.y * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);
								illumination.z = illumination.z + KS.z * lights[l].Specular.z * pow(dotProduct(vectMult(Eye, -1), reflectiveRay), 1000);

								illumination.x = illumination.x + KA.x * lights[l].Ambient.x;
								illumination.y = illumination.y + KA.y * lights[l].Ambient.y;
								illumination.z = illumination.z + KA.z * lights[l].Ambient.z;
								//printf("Ambient: %f, %f, %f :\n", illumination.x, illumination.y, illumination.z);
								//illumination = vectAdd(vectAdd(diffIllum, specIllum), ambIllum);
							}
						}
					}
					//printf("%f, %f, %f :\n", illumination.x, illumination.y, illumination.z);
					// Assign it to array that stores all the pixels to be colored
					colorArray[i][j] = illumination;
				}
			}
		}
}

/***************************************************************************/
void init_window()
/* Clear the image area, and set up the coordinate system */
{

	/* Clear the window */
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glOrtho(0, WIDTH, 0, HEIGHT, -1, 1);
}

/***************************************************************************/

void write_pixel(int x, int y, double r, double g, double b)
/* Turn on the pixel found at x,y */
{

	glColor3f(r, g, b);
	glBegin(GL_POINTS);
	glVertex3i(x, y, 0);
	glEnd();
}

//***************************************************************************/
// Midpoint algorithm to draw lines
void midPoint(int x0, int y0, int x1, int y1) {
	float m = (y1 - y0);
	m = m / (x1 - x0);
	if ((y1 == y0) || (x1 == x0)) {
		m = 0;
	}

	if (m == 0) {
		if (x1 == x0 && y0 < y1) {
			int x = x0;
			int y = y0;
			for (y = y0; y < y1; y++) {
				write_pixel(x, y, 1.0, 1.0, 1.0);
			}
		}

		else if (x1 == x0 && y1 < y0) {
			int x = x0;
			int y = y1;
			for (y = y1; y < y0; y++) {
				write_pixel(x, y, 1.0, 1.0, 1.0);
			}
		}

		else if (y1 == y0 && x0 < x1) {
			int x = x0;
			int y = y0;
			for (x = x0; x < x1; x++) {
				write_pixel(x, y, 1.0, 1.0, 1.0);
			}
		}

		else if (y1 == y0 && x1 < x0) {
			int x = x1;
			int y = y0;
			for (x = x1; x < x0; x++) {
				write_pixel(x, y, 1.0, 1.0, 1.0);
			}
		}
	}

	if (x0 < x1 && y0 < y1) {
		if (m <= 1 && m > 0) {
			int X = x0;
			int Y = y0;
			int a = y1 - y0;
			int b = -(x1 - x0);
			float d = a + b / 2;
			for (X = x0; X < x1; X++) {
				write_pixel(X, Y, 1.0, 1.0, 1.0);
				if (d < 0) {
					d = d + a;
				}
				else {
					//Y++;
					d = d + a + b;
					Y++;
				}
			}
		}

		else if (m >= 1) {
			int X = x0;
			int Y = y0;
			int a = y1 - y0;
			int b = x1 - x0;
			float d = b - (a / 2);
			for (Y = y0; Y < y1; Y++) {
				write_pixel(X, Y, 1.0, 1.0, 1.0);
				if (d < 0) {
					d = d + b;
				}
				else {
					X++;
					d = d + (b - a);
				}
			}
		}
	}

	if (x1 < x0 && y1 < y0) {
		if (m <= 1 && m > 0) {
			int X = x1;
			int Y = y1;
			int a = y1 - y0;
			int b = -(x1 - x0);
			float d = -a - b / 2;
			for (X = x1; X < x0; X++) {
				write_pixel(X, Y, 1.0, 1.0, 1.0);
				if (d < 0) {
					d = d - a;
				}
				else {
					//Y++;
					d = d - a - b;
					Y++;
				}
			}
		}

		else if (m >= 1) {
			int X = x0;
			int Y = y0;
			int a = y1 - y0;
			int b = -(x1 - x0);
			//printf("a, b is (%d,%d)\n", a, b);
			float d = -b - (a / 2);
			//printf("d is (%f)\n", d);
			for (Y = y0; Y >= y1; --Y) {
				write_pixel(X, Y, 1.0, 1.0, 1.0);
				if (d < 0) {
					d = d + b;
				}
				else {
					d = d + (b + a);
					--X;
				}
			}
		}
	}

	else if (y0 > y1&& x0 < x1) {
		if (m < 0 && m >= -1) {
			int X = x1; // mod
			int Y = y1; // mod
			int a = y1 - y0; //dy
			int b = -(x1 - x0); // -dx
			float d = -a + b / 2;
			for (X = x1 - 1; X >= x0; --X) {
				write_pixel(X, Y, 1.0, 1.0, 1.0);
				if (d < 0) {
					d = d - a;
				}
				else {
					//Y++;
					d = d - a + b;
					Y++;
				}
			}

		}

		else if (m <= -1) {
			int X = x0; // mod
			int Y = y0; // mod
			int a = y1 - y0;
			int b = -(x1 - x0);
			float d = -b + (a / 2);
			for (Y = y0; Y >= y1; Y--) {

				write_pixel(X, Y, 1.0, 1.0, 1.0);
				if (d < 0) {
					d = d - b;
				}
				else {
					X++;
					d = d - (b - a);
				}
			}
		}
	}

	else if (x0 > x1&& y0 < y1) {
		if (m < 0 && m >= -1) {
			int X = x0; // mod
			int Y = y0; // mod
			int a = y1 - y0; //dy
			int b = -(x1 - x0); // -dx
			float d = a - b / 2;
			for (X = x0; X >= x1; X--) {
				write_pixel(X, Y, 1.0, 1.0, 1.0);
				if (d < 0) {
					d = d + a;
				}
				else {
					//Y++;
					d = d + a - b;
					Y++;
				}
			}

		}

		else if (m <= -1) {
			int X = x1; // mod
			int Y = y1; // mod
			int a = y1 - y0;
			int b = x1 - x0;
			float d = b + (a / 2);
			for (Y = y1 - 1; Y >= y0; --Y) {
				write_pixel(X, Y, 1.0, 1.0, 1.0);
				if (d < 0) {
					d = d - b;
				}
				else {
					X++;
					d = d - (b + a);
				}
			}
		}
	}
}
void clear_screen()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void swap_buffer()
{
	printf("buffer swap \n");
	glutSwapBuffers();
}
// function to return to center points of the model to center it when using transformation matrix
verticesStruct getCenterVertices() {
	float xSum = 0.0, ySum = 0.0, zSum = 0.0, wSum = 0.0, maxValue = 0.0, minValue = 0.0;
	float centerX, centerY, centerZ, centerW;
	int vertexSize = MainVertices.size();

	verticesStruct CVertices;

	for (size_t i = 0; i < MainVertices.size(); ++i) {
		xSum += MainVertices[i].x;
		ySum += MainVertices[i].y;
		zSum += MainVertices[i].z;
		wSum += MainVertices[i].w;
	}
	centerX = xSum / vertexSize;
	centerY = ySum / vertexSize;
	centerZ = zSum / vertexSize;
	centerW = wSum / vertexSize;

	CVertices.x = centerX;
	CVertices.y = centerY;
	CVertices.z = centerY;

	return CVertices;
}

// Function to multiple 4 by 4 matrix with a 4 by 1 matrix
void matrixMult(float iMat[4][4], int indx) {
	float inputvert[4][1] = { {MainVertices[indx].x}, {MainVertices[indx].y}, {MainVertices[indx].z}, {1.0} };
	float outputvert[4][1] = { { 0.0}, {0.0}, {0.0}, {0.0} };
	int k = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 1; ++j) {
			for (int k = 0; k < 4; ++k)
			{
				outputvert[i][j] += iMat[i][k] * inputvert[k][j];
			}
		}
	}
	MainVertices[indx].x = outputvert[0][0];
	MainVertices[indx].y = outputvert[1][0];
	MainVertices[indx].z = outputvert[2][0];
}

// function to determine the type of transformation and performing the,
void detectTransform(int transformType, char direc) {
	// Main matrices to do composite transformation multiplication
	float identityMat[16] =
	{
		1.0, 0, 0, 0,
		0, 1.0, 0, 0,
		0, 0, 1.0, 0,
		0, 0, 0, 1.0
	};
	float CompositeMatrix[4][4]
		= {
			{1.0, 0, 0, 0},
			{0, 1.0, 0, 0},
			{0, 0, 1.0, 0},
			{0, 0, 0, 1.0}
	};
	float InvCompositeMatrix[4][4]
		= {
			{1.0, 0, 0, 0},
			{0, 1.0, 0, 0 },
			{0, 0, 1.0, 0},
			{0, 0, 0, 1.0}
	};
	float GlobalCompositeMatrix[4][4]
		= {
			{0, 0, 0, 0},
			{0, 0, 0, 0},
			{0, 0, 0, 0},
			{0, 0, 0, 0}
	};
	float iMatrx[4][4] =
	{
		{1.0, 0, 0, 0},
		{0, 1.0, 0, 0},
		{0, 0, 1.0, 0},
		{0, 0, 0, 1.0}
	};

	float tempCompMat[4][4] =
	{
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0}
	};
	if (transformType == 1) { // transform is translation
		if (direc == 'a')
			iMatrx[0][3] = -10;
		else if (direc == 'd')
			iMatrx[0][3] = 10;
		else if (direc == 'w')
			iMatrx[1][3] = 10;
		else if (direc == 's')
			iMatrx[1][3] = -10;
		// Translate the model
		for (size_t i = 0; i < MainVertices.size(); i++)
			matrixMult(iMatrx, i);
	}
	else if (transformType == 2) { // transform is scaling
		verticesStruct tempVertices;
		if (direc == 'a')
			iMatrx[0][0] = 1.05, iMatrx[1][1] = 1.05, iMatrx[2][2] = 1.05;
		else if (direc == 'd')
			iMatrx[0][0] = 1.0 / 1.05, iMatrx[1][1] = 1.0 / 1.05, iMatrx[2][2] = 1.0 / 1.05;
		else if (direc == 'w')
			iMatrx[0][0] = 1.05, iMatrx[1][1] = 1.05, iMatrx[2][2] = 1.05;
		else if (direc == 's')
			iMatrx[0][0] = 1.0 / 1.05, iMatrx[1][1] = 1.0 / 1.05, iMatrx[2][2] = 1.0 / 1.05;
		// center vertices
		tempVertices = getCenterVertices();
		// composite multiplication of center, translation matrix, scaling, and inverse translation matrices
		CompositeMatrix[0][3] = tempVertices.x, CompositeMatrix[1][3] = tempVertices.y, CompositeMatrix[2][3] = tempVertices.z;
		InvCompositeMatrix[0][3] = -1.0 * (tempVertices.x), InvCompositeMatrix[1][3] = -1.0 * (tempVertices.y), InvCompositeMatrix[2][3] = -1.0 * (tempVertices.z);
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				for (int k = 0; k < 4; ++k)
				{
					tempCompMat[i][j] += iMatrx[i][k] * CompositeMatrix[k][j];
				}
			}
		}
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				for (int k = 0; k < 4; ++k)
				{
					GlobalCompositeMatrix[i][j] += InvCompositeMatrix[i][k] * tempCompMat[k][j];
				}
			}
		}
		// Multiply final composite matrix with the vertex points
		for (size_t i = 0; i < MainVertices.size(); i++)
			matrixMult(GlobalCompositeMatrix, i);
		// Finally center the model
		centerVertices();
	}

	else if (transformType == 3) { // transform is rotating
		verticesStruct tempVertices;
		if (direc == 'a')
			iMatrx[0][0] = cos(7 * 3.14 / 180), iMatrx[0][2] = sin(7 * 3.14 / 180), iMatrx[2][0] = -sin(7 * 3.14 / 180), iMatrx[2][2] = cos(7 * 3.14 / 180);
		else if (direc == 'd')
			iMatrx[0][0] = cos(-7 * 3.14 / 180), iMatrx[0][2] = sin(-7 * 3.14 / 180), iMatrx[2][0] = -sin(-7 * 3.14 / 180), iMatrx[2][2] = cos(-7 * 3.14 / 180);
		else if (direc == 'w')
			iMatrx[1][1] = cos(7 * 3.14 / 180), iMatrx[1][2] = -sin(7 * 3.14 / 180), iMatrx[2][1] = sin(7 * 3.14 / 180), iMatrx[2][2] = cos(7 * 3.14 / 180);
		else if (direc == 's')
			iMatrx[1][1] = cos(-7 * 3.14 / 180), iMatrx[1][2] = -sin(-7 * 3.14 / 180), iMatrx[2][1] = sin(-7 * 3.14 / 180), iMatrx[2][2] = cos(-7 * 3.14 / 180);
		tempVertices = getCenterVertices();

		// composite multiplication of center, translation matrix, rotation, and inverse translation matrices
		CompositeMatrix[0][3] = tempVertices.x, CompositeMatrix[1][3] = tempVertices.y, CompositeMatrix[2][3] = tempVertices.z;
		InvCompositeMatrix[0][3] = -1.0 * (tempVertices.x), InvCompositeMatrix[1][3] = -1.0 * (tempVertices.y), InvCompositeMatrix[2][3] = -1.0 * (tempVertices.z);
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				for (int k = 0; k < 4; ++k)
				{
					tempCompMat[i][j] += iMatrx[i][k] * CompositeMatrix[k][j];
				}
			}
		}
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				for (int k = 0; k < 4; ++k)
				{
					GlobalCompositeMatrix[i][j] += InvCompositeMatrix[i][k] * tempCompMat[k][j];
				}
			}
		}
		// Final composite matrix multiplication of vertex points
		for (size_t i = 0; i < MainVertices.size(); i++) {
			matrixMult(GlobalCompositeMatrix, i);
		}
		// center the model
		centerVertices();
	}
}

// Function to assign the w vertex of the model, gets called when perspective view is called
// performs and produces matrix with z/d
void view(int indx) {
	float inputvert[4] = { MainVertices[indx].x, MainVertices[indx].y, MainVertices[indx].z, 1 };
	float outputvert[4] = { 0, 0, 0, 0 };
	int k = 0;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			outputvert[i] = outputvert[i] + presMat[k++] * inputvert[j];
		}
	}
	//std::cout << "W: " << outputvert[3] << std::endl;
	MainVertices[indx].w = outputvert[3];
}

// If view is orthogonal then set w to 1
void viewOrth(int indx) {
	MainVertices[indx].w = 1;
}

void display(void)   // Create The Display Function
{

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	      // Clear Screen 

	write_pixel(x_last, y_last, 1.0, 1.0, 1.0);//<-you can get rid of this call if you like
	// CALL YOUR CODE HERE
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			write_pixel(i, j, colorArray[i][j].x, colorArray[i][j].y, colorArray[i][j].z);
			//printf("I and J: %d, %d\n", i, j);
		}
	}
	

	glutSwapBuffers();                                      // Draw Frame Buffer 
}

/***************************************************************************/
void mouse(int button, int state, int x, int y)
{
	/* This function I finessed a bit, the value of the printed x,y should
	   match the screen, also it remembers where the old value was to avoid multiple
	   readings from the same mouse click.  This can cause problems when trying to
	   start a line or curve where the last one ended */
	static int oldx = 0;
	static int oldy = 0;
	int mag;

	//y *= -1;  //align y with mouse
	//y += 500; //ignore 
	mag = (oldx - x) * (oldx - x) + (oldy - y) * (oldy - y);
	if (mag > 20) {
		printf(" x,y is (%d,%d)\n", x, y);
	}
	oldx = x;
	oldy = y;
	x_last = x;
	y_last = y;
}

/***************************************************************************/
void keyboard(unsigned char key, int x, int y)  // Create Keyboard Function
{

	switch (key) {
	case 27:              // When Escape Is Pressed...
		exit(0);   // Exit The Program
		break;
	case '1':             // stub for new screen
		printf("New screen\n");
		break;
	case 'v':
		std::cout << "view switch: " << viewSwitch << std::endl;
		if (viewSwitch == false) {
			// When view is perspective then center the model for the initial view
			for (size_t i = 0; i < MainVertices.size(); i++) {
				MainVertices[i].x = MainVertices[i].x - centerX;
				MainVertices[i].y = MainVertices[i].y - centerY;
				MainVertices[i].z = MainVertices[i].z - centerZ + maxValue + minValue;
				MainVertices[i].w = MainVertices[i].w - centerW;
			}
			viewSwitch = true;
		}
		break;
	case 't':             // translate left
		translateSwitch = 1;
		scaleSwitch = 0;
		rotSwitch = 0;
		printf("Translating\n");
		break;
	case 'e':             // translate left
		scaleSwitch = 1;
		translateSwitch = 0;
		rotSwitch = 0;
		printf("Scaling\n");
		break;
	case 'r':             // translate left
		rotSwitch = 1;
		translateSwitch = 0;
		scaleSwitch = 0;
		printf("Rotating\n");
		break;
		// wasd direction transformation
	case 'a':
		if (translateSwitch == 1)
			detectTransform(1, 'a');
		else if (scaleSwitch == 1)
			detectTransform(2, 'a');
		else if (rotSwitch == 1)
			detectTransform(3, 'a');
		break;
	case 'd':
		if (translateSwitch == 1)
			detectTransform(1, 'd');
		else if (scaleSwitch == 1)
			detectTransform(2, 'd');
		else if (rotSwitch == 1)
			detectTransform(3, 'd');
		break;
	case 'w':
		if (translateSwitch == 1)
			detectTransform(1, 'w');
		else if (scaleSwitch == 1)
			detectTransform(2, 'w');
		else if (rotSwitch == 1)
			detectTransform(3, 'w');
		break;
	case 's':
		if (translateSwitch == 1)
			detectTransform(1, 's');
		else if (scaleSwitch == 1)
			detectTransform(2, 's');
		else if (rotSwitch == 1)
			detectTransform(3, 's');
		break;
	case 'p':
		swap_buffer();
		break;
	case 'l':
		rayTrace();
		break;
	case 'o':
		clear_screen();
		break;
	case 'b':
		printf("Bump Mapping:\n");
		BumpMapON = !BumpMapON;
		clear_screen();
		rayTrace();
		swap_buffer();
		break;
	case 'x':
		printf("Phong Smooth:\n");
		PhongSmoothON = !PhongSmoothON;
		clear_screen();
		rayTrace();
		swap_buffer();
		break;
	default:
		break;
	}
}
/***************************************************************************/

int main(int argc, char* argv[])
{
	/* This main function sets up the main loop of the program and continues the
	   loop until the end of the data is reached.  Then the window can be closed
	   using the escape key.						  */

	// Assigning Light values for the scene
	Light l, j;
	l.Position.x = -300;
	l.Position.y = 300;
	l.Position.z = -100;
	l.Ambient.x = 0.1; // 0.2
	l.Ambient.y = 0.1; // 0.44
	l.Ambient.z = 0.1;
	l.Specular.x = 1; // 0.15
	l.Specular.y = 1; // 0.2
	l.Specular.z = 1; // 0.15
	l.Diffuse.x = 0.43; // 0.15
	l.Diffuse.y = 0.21; // 0.25
	l.Diffuse.z = 0.79; //0.31

	//j.Position.x = 170;
	//j.Position.y = 170;
	//j.Position.z = -20;
	//j.Ambient.x = 0.04;
	//j.Ambient.y = 0.196;
	//j.Ambient.z = 0.125;
	//j.Specular.x = 0.04;
	//j.Specular.y = 0.196;
	//j.Specular.z = 0.125;
	//j.Diffuse.x = 0.04;
	//j.Diffuse.y = 0.196;
	//j.Diffuse.z = 0.125;
	lights.push_back(l);
	//lights.push_back(j);

	readObjFile("model2.obj", vertices, uvs, normals);
	// normalize the model
	normalizeVertices();
	//centerVertices();
	rayTrace();
	//centerVertices();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Computer Graphics");
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);

	init_window();				             //create_window

	glutMainLoop();                 // Initialize The Main Loop
}