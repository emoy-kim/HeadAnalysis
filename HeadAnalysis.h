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

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <chrono>
#include <filesystem>
#include "ModelRender.h"

class HeadAnalysis
{
public:
   HeadAnalysis(const HeadAnalysis&) = delete;
   HeadAnalysis(const HeadAnalysis&&) = delete;
   HeadAnalysis& operator=(const HeadAnalysis&) = delete;
   HeadAnalysis& operator=(const HeadAnalysis&&) = delete;

   HeadAnalysis();
   ~HeadAnalysis() = default;

   void frontalize(cv::Mat& frontalized, const cv::Mat& query);
   void lateralize(cv::Mat& lateralized, const cv::Mat& query, const cv::Vec3f& euler_angle_in_degree);
   void estimateHeadPosition(cv::Matx31f& euler_angle_in_degree, const cv::Mat& query);

private:
   enum class OCCLUSION { NO=0, LEFT, RIGHT };

   std::chrono::time_point<std::chrono::system_clock> StartTime;

   int ScreenWidth;
   int ScreenHeight;

   dlib::frontal_face_detector FaceDetector;
   dlib::shape_predictor LandmarkDetector;

   ModelRender ModelRenderer;
   cv::Matx44f ClipToCameraMatrix;
   cv::Mat ModelDepth;
   std::vector<cv::Point3f> Reference3DLandmarks;

   void tic() { StartTime = std::chrono::system_clock::now(); }
   void tok(std::string&& function_name) const
   {
      const std::chrono::duration<double> time = (std::chrono::system_clock::now() - StartTime) * 1000.0;
      std::cout << function_name << ": " << time.count() << " ms\n";
   }

   void get3DModelLandmarks(std::vector<cv::Point3f>& model_3d_landmarks, const cv::Vec3f& euler_angle_in_degree);
   bool get3DCoordinates(cv::Point3f& unprojected_3d, const cv::Point2f& projected_2d);

   static bool isValidLandmark(int landmark_index);
   bool detectFaceLandmarks(std::vector<cv::Point2f>& landmarks, const cv::Mat& image);
   void estimateCameraPose(
      cv::Matx33f& rotation, 
      cv::Matx34f& projection, 
      const cv::Mat& query,
      const std::vector<cv::Point3f>& landmarks_3d
   );
   void getMapperToQuery(cv::Mat& mapper_to_query, const cv::Matx34f& projection, const cv::Size& query_size);
   void getBilinearInterpolatedColor(cv::Vec3f& bgr_color, const cv::Mat& image, const cv::Vec2f& point) const;
   void estimateVirtualFace(cv::Mat& virtualized, const cv::Mat& query, const cv::Mat& mapper_to_query) const;
   void calculateReferenceNumberOfQuery(
      cv::Mat& reference_number_of_query, 
      const cv::Mat& mapper_to_query, 
      const cv::Mat& query
   ) const;
   OCCLUSION getIndicatorIfOccluded(cv::Mat& occlusion_indicator, const cv::Mat& reference_number_of_query) const;
   void applySoftSymmetry(
      cv::Mat& soft_symmetry, 
      const cv::Mat& initial_estimated, 
      const cv::Mat& reference_number_of_query,
      const cv::Mat& occlusion_indicator
   ) const;
   void calculateVisibility(const cv::Mat& reference_number_of_query) const;

   void drawHeadPosition(const cv::Vec3f& euler_angle_in_radian, const cv::Mat& query ) const;
};