/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is a free software; it can be freely used, changed and redistributed.
 * If you use any version of the code, please reference the code.
 * 
 * 
 * NOTE:
 *   1) _frontalize()_ function is based on the paper in [1].
 *     - In [1], they support their own codes, but I refactored and modified a little.
 *   
 *   
 * [1] https://talhassner.github.io/home/publication/2015_CVPR_1
 * 
 */

#pragma once

#include <OpenCVLinker.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <chrono>
#include "ModelRender.h"

using namespace std;
using namespace cv;
using namespace dlib;
using namespace chrono;

class HeadAnalysis
{
	enum { NO_OCCLUSION=0, LEFT_OCCLUSION, RIGHT_OCCLUSION };

	time_point<system_clock> StartTime;

	int ScreenWidth;
	int ScreenHeight;

	frontal_face_detector FaceDetector;
	shape_predictor LandmarkDetector;

	ModelRender ModelRenderer;
	Matx44f ClipToCameraMatrix;
	Mat ModelDepth;
	std::vector<Point3f> Reference3DLandmarks;

	void tic() { StartTime = system_clock::now(); }
	void tok(string&& function_name) const
	{
		const duration<double> time = (system_clock::now() - StartTime) * 1000.0;
		cout << function_name << ": " << time.count() << " ms" << endl;
	}

	void get3DModelLandmarks(std::vector<Point3f>& model_3d_landmarks, const Vec3f& euler_angle_in_degree);
	bool get3DCoordinates(Point3f& unprojected_3d, const Point2f& projected_2d);

	bool isValidLandmark(const int& landmark_index) const;
	bool detectFaceLandmarks(std::vector<Point2f>& landmarks, const Mat& image);
	void estimateCameraPose(
		Matx33f& rotation, 
		Matx34f& projection, 
		const Mat& query,
		const std::vector<Point3f>& landmarks_3d
	);
	void getMapperToQuery(Mat& mapper_to_query, const Matx34f& projection, const Size& query_size);
	void getBilinearInterpolatedColor(Vec3f& bgr_color, const Mat& image, const Vec2f& point) const;
	void estimateVirtualFace(Mat& virtualized, const Mat& query, const Mat& mapper_to_query) const;
	void calculateReferenceNumberOfQuery(
		Mat& reference_number_of_query, 
		const Mat& mapper_to_query, 
		const Mat& query
	) const;
	int getIndicatorIfOccluded(Mat& occlusion_indicator, const Mat& reference_number_of_query) const;
	void applySoftSymmetry(
		Mat& soft_symmetry, 
		const Mat& initial_estimated, 
		const Mat& reference_number_of_query,
		const Mat& occlusion_indicator
	) const;
	void calculateVisibility(const Mat& reference_number_of_query) const;

	void drawHeadPosition(const Vec3f& euler_angle_in_radian, const Mat& query ) const;


public:
	HeadAnalysis(const HeadAnalysis&) = delete;
	HeadAnalysis(const HeadAnalysis&&) = delete;
	HeadAnalysis& operator=(const HeadAnalysis&) = delete;
	HeadAnalysis& operator=(const HeadAnalysis&&) = delete;

	HeadAnalysis();
	~HeadAnalysis() = default;

	void frontalize(Mat& frontalized, const Mat& query);
	void lateralize(Mat& lateralized, const Mat& query, const Vec3f& euler_angle_in_degree);
	void estimateHeadPosition(Matx31f& euler_angle_in_degree, const Mat& query);
};