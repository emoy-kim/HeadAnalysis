#include "HeadAnalysis.h"

HeadAnalysis::HeadAnalysis() : ScreenWidth( 500 ), ScreenHeight( 500 )
{
   FaceDetector = dlib::get_frontal_face_detector();

   const std::string model_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples/model";
   dlib::deserialize( model_directory_path + "/shape_predictor_68_face_landmarks.dat" ) >> LandmarkDetector;

   get3DModelLandmarks( Reference3DLandmarks, cv::Vec3f(0.0f, 0.0f, 0.0f) );
}

void HeadAnalysis::get3DModelLandmarks(std::vector<cv::Point3f>& model_3d_landmarks, const cv::Vec3f& euler_angle_in_degree)
{
   cv::Mat model_face;
   ModelRenderer.getModelImage( model_face, ModelDepth, 500, 500, 500.0f, euler_angle_in_degree );
   ModelRenderer.getClipToCameraMatrix( ClipToCameraMatrix );

   std::vector<cv::Point2f> model_landmarks;
   if (!detectFaceLandmarks( model_landmarks, model_face )) {
      std::cout << ">> Could not find landmarks from the model...\n";
      return;
   }

   model_3d_landmarks.clear();
   for (const auto& landmark : model_landmarks) {
      cv::Point3f unprojected;
      get3DCoordinates( unprojected, landmark );
      model_3d_landmarks.emplace_back( unprojected );
   }
}

bool HeadAnalysis::get3DCoordinates(cv::Point3f& unprojected_3d, const cv::Point2f& projected_2d)
{
   const cv::Point screen(static_cast<int>(round( projected_2d.x )), static_cast<int>(round( projected_2d.y )));
   const auto depth = ModelDepth.at<float>(screen.y, screen.x);
   if (depth == DEPTH_INFINITY) return false;
   
   const cv::Point2f screen_in_opencv_system(projected_2d.x, static_cast<float>(ModelDepth.rows) - projected_2d.y);
   const cv::Matx41f ndc_point(
      2.0f * screen_in_opencv_system.x / static_cast<float>(ModelDepth.cols) - 1.0f,
      (2.0f * screen_in_opencv_system.y / static_cast<float>(ModelDepth.rows) - 1.0f), 
      2.0f * depth - 1.0f,
      1.0f
   );
   
   const cv::Matx41f camera_point = ClipToCameraMatrix * ndc_point;
   unprojected_3d.x = camera_point(0) / camera_point(3);
   unprojected_3d.y = camera_point(1) / camera_point(3);
   unprojected_3d.z = camera_point(2) / camera_point(3);
   return true;
}

bool HeadAnalysis::isValidLandmark(int landmark_index)
{
   // jaw line excluded
   return landmark_index >= 17;
}

bool HeadAnalysis::detectFaceLandmarks(std::vector<cv::Point2f>& landmarks, const cv::Mat& image)
{
   dlib::array2d<dlib::rgb_pixel> dlib_image;
   assign_image( dlib_image, dlib::cv_image<dlib::bgr_pixel>(image) );
   std::vector<dlib::rectangle> faces = FaceDetector(dlib_image);
   if (faces.empty()) {
      std::cout << ">> The image must contain at least one person's face!\n";
      return false;
   }

   // only use the first detected person
   dlib::full_object_detection estimated_landmarks = LandmarkDetector(dlib_image, faces[0]);
   
   landmarks.clear();
   for (uint i = 0; i < estimated_landmarks.num_parts(); ++i) {
      if (!isValidLandmark( i )) continue;
      landmarks.emplace_back( estimated_landmarks.part(i).x(), estimated_landmarks.part(i).y() );
   }
   return true;
}

void HeadAnalysis::estimateCameraPose(
   cv::Matx33f& rotation, 
   cv::Matx34f& projection, 
   const cv::Mat& query, 
   const std::vector<cv::Point3f>& landmarks_3d
)
{
   std::vector<cv::Point2f> landmarks;
   if (!detectFaceLandmarks( landmarks, query )) {
      std::cout << ">> Could not find landmarks from the query...\n";
      return;
   }

   const auto focal_length = static_cast<float>(std::max( query.rows, query.cols ));
   const cv::Matx33f intrinsic(
      focal_length, 0.0f, static_cast<float>(query.cols) * 0.5f,
      0.0f, focal_length, static_cast<float>(query.cols) * 0.5f,
      0.0f, 0.0f, 1.0f
   );

   cv::Mat rotation_vector(3, 1, CV_32FC1);
   cv::Mat translation_vector(3, 1, CV_32FC1);
   solvePnP(
      landmarks_3d, 
      landmarks, 
      cv::Mat(intrinsic), 
      cv::Mat(), 
      rotation_vector, 
      translation_vector, 
      cv::SOLVEPNP_ITERATIVE
   );
   cv::Matx31f translation = translation_vector;
   cv::Mat rotation_matrix;
   Rodrigues( rotation_vector, rotation_matrix );
   rotation = rotation_matrix;

   projection = intrinsic * cv::Matx34f(
      rotation(0, 0), rotation(0, 1), rotation(0, 2), translation(0),
      rotation(1, 0), rotation(1, 1), rotation(1, 2), translation(1),
      rotation(2, 0), rotation(2, 1), rotation(2, 2), translation(2)
   );
}

void HeadAnalysis::getMapperToQuery(cv::Mat& mapper_to_query, const cv::Matx34f& projection, const cv::Size& query_size)
{
   mapper_to_query.create( ScreenHeight, ScreenWidth, CV_32FC2 );

   const auto width_boundary = static_cast<float>(query_size.width - 1);
   const auto height_boundary = static_cast<float>(query_size.height - 1);
   for (int j = 0; j < mapper_to_query.rows; ++j) {
      auto* mapper_ptr = mapper_to_query.ptr<cv::Vec2f>(j);
      for (int i = 0; i < mapper_to_query.cols; ++i) {
         cv::Point3f unprojected;
         cv::Point2f screen_point(static_cast<float>(i), static_cast<float>(j));
         if (get3DCoordinates( unprojected, screen_point )) {
            cv::Matx31f query_point = projection * cv::Matx41f(unprojected.x, unprojected.y, unprojected.z, 1.0f);
            query_point(0) /= query_point(2);
            query_point(1) /= query_point(2);
            mapper_ptr[i](0) = std::min( std::max( 0.0f, query_point(0) ), width_boundary );
            mapper_ptr[i](1) = std::min( std::max( 0.0f, query_point(1) ), height_boundary );
         }
         else mapper_ptr[i] = -1.0f;
      }
   }
}

void HeadAnalysis::getBilinearInterpolatedColor(
   cv::Vec3f& bgr_color, 
   const cv::Mat& image, 
   const cv::Vec2f& point
) const
{
   const int x0 = static_cast<int>(floor( point(0) ));
   const int y0 = static_cast<int>(floor( point(1) ));
   const int x1 = std::min( x0 + 1, image.cols - 1 );
   const int y1 = std::min( y0 + 1, image.rows - 1 );
   const auto tx = point(0) - static_cast<float>(x0);
   const auto ty = point(1) - static_cast<float>(y0);

   const auto* curr_row = image.ptr<cv::Vec3b>(y0);
   const auto* next_row = image.ptr<cv::Vec3b>(y1);
   bgr_color = 
      static_cast<cv::Vec3f>(curr_row[x0]) * (1.0f - tx) * (1.0f - ty) + 
      static_cast<cv::Vec3f>(curr_row[x1]) * tx * (1.0f - ty) + 
      static_cast<cv::Vec3f>(next_row[x0]) * (1.0f - tx) * ty + 
      static_cast<cv::Vec3f>(next_row[x1]) * tx * ty;
}

void HeadAnalysis::estimateVirtualFace(cv::Mat& virtualized, const cv::Mat& query, const cv::Mat& mapper_to_query) const
{
   virtualized.create( ScreenHeight, ScreenWidth, CV_32FC3 );
   for (int j = 0; j < virtualized.rows; ++j) {
      const auto* mapper_ptr = mapper_to_query.ptr<cv::Vec2f>(j);
      auto* estimated_ptr = virtualized.ptr<cv::Vec3f>(j);
      for (int i = 0; i < virtualized.cols; ++i) {
         if (mapper_ptr[i](0) < 0.0f) {
            estimated_ptr[i] = cv::Vec3f(255.0f, 255.0f, 255.0f);
         }
         else {
            cv::Vec3f bgr_color;
            getBilinearInterpolatedColor( bgr_color, query, mapper_ptr[i] );
            estimated_ptr[i] = bgr_color;
         }
      }
   }
}

void HeadAnalysis::calculateReferenceNumberOfQuery(
   cv::Mat& reference_number_of_query, 
   const cv::Mat& mapper_to_query, 
   const cv::Mat& query
) const
{
   cv::Mat number_of_being_referenced = cv::Mat::zeros(query.size(), CV_32SC1);
   for (int j = 0; j < mapper_to_query.rows; ++j) {
      const auto* mapper_ptr = mapper_to_query.ptr<cv::Vec2f>(j);
      for (int i = 0; i < mapper_to_query.cols; ++i) {
         if (mapper_ptr[i](0) >= 0.0f) {
            const int x = static_cast<int>(round( mapper_ptr[i](0) ));
            const int y = static_cast<int>(round( mapper_ptr[i](1) ));
            const int data_index = y * query.cols + x;
            reinterpret_cast<int*>(number_of_being_referenced.data)[data_index]++;
         }
      }
   }

   reference_number_of_query = cv::Mat::zeros(mapper_to_query.size(), CV_32SC1);
   for (int j = 0; j < mapper_to_query.rows; ++j) {
      const auto* mapper_ptr = mapper_to_query.ptr<cv::Vec2f>(j);
      auto* reference_ptr = reference_number_of_query.ptr<int>(j);
      for (int i = 0; i < mapper_to_query.cols; ++i) {
         if (mapper_ptr[i](0) >= 0.0f) {
            const int x = static_cast<int>(round( mapper_ptr[i](0) ));
            const int y = static_cast<int>(round( mapper_ptr[i](1) ));
            const int data_index = y * query.cols + x;
            reference_ptr[i] = reinterpret_cast<int*>(number_of_being_referenced.data)[data_index];
         }
      }
   }

   reference_number_of_query.convertTo( reference_number_of_query, CV_32FC1 );
   GaussianBlur(
      reference_number_of_query, 
      reference_number_of_query, 
      cv::Size(15, 15), 
      30.0, 
      cv::BORDER_REPLICATE
   );
}

HeadAnalysis::OCCLUSION HeadAnalysis::getIndicatorIfOccluded(cv::Mat& occlusion_indicator, const cv::Mat& reference_number_of_query) const
{
   const cv::Mat& reference = reference_number_of_query;
   const auto total_sum = round( cv::sum( reference )[0] );
   const int middle = reference.cols / 2;
   const cv::Rect left_part(0, 0, middle, reference.rows);
   const cv::Rect right_part(middle, 0, reference.cols - middle, reference.rows);
   const auto left_sum = static_cast<int>(round( cv::sum( reference(left_part) )[0] ));
   const auto right_sum = static_cast<int>(round( cv::sum( reference(right_part) )[0] ));
   const int sum_difference = left_sum - right_sum;

   if (fabs( sum_difference ) * 4.5 < total_sum) return OCCLUSION::NO;

   const OCCLUSION occlusion_position = sum_difference > 0 ? OCCLUSION::LEFT : OCCLUSION::RIGHT;
   occlusion_indicator = cv::Mat::ones(reference.size(), CV_32FC1);
   switch (occlusion_position) {
      case OCCLUSION::LEFT:
         occlusion_indicator(left_part) = 0.0f;
         break;
      case OCCLUSION::RIGHT:
         occlusion_indicator(right_part) = 0.0f;
         break;
      default: break;
   }
   GaussianBlur(
      occlusion_indicator, 
      occlusion_indicator, 
      cv::Size(15, 15), 
      60.0, 
      cv::BORDER_REPLICATE
   );
   return occlusion_position;
}

void HeadAnalysis::applySoftSymmetry(
   cv::Mat& soft_symmetry, 
   const cv::Mat& initial_estimated, 
   const cv::Mat& reference_number_of_query,
   const cv::Mat& occlusion_indicator
) const
{
   double normalizer;
   minMaxLoc( reference_number_of_query, nullptr, &normalizer, nullptr, nullptr );
   normalizer = 1.0 / normalizer;

   soft_symmetry.create( initial_estimated.size(), initial_estimated.type() );
   for (int j = 0; j < soft_symmetry.rows; ++j) {
      const auto* initial_ptr = initial_estimated.ptr<cv::Vec3f>(j);
      const auto* reference_ptr = reference_number_of_query.ptr<float>(j);
      const auto* indicator_ptr = occlusion_indicator.ptr<float>(j);
      auto* symmetry_ptr = soft_symmetry.ptr<cv::Vec3f>(j);
      
      for (int i = 0; i < soft_symmetry.cols; ++i) {
         const int symmetric_index = soft_symmetry.cols - 1 - i;
         const float visibility_factor = 1.0f / exp( 0.5f + reference_ptr[i] * static_cast<float>(normalizer) );
         const float initial_color_weight = indicator_ptr[i] + visibility_factor * indicator_ptr[symmetric_index];
         const float symmetric_color_weight = (1.0f - visibility_factor) * indicator_ptr[symmetric_index];

         const cv::Vec3f initial_color = initial_ptr[i];
         const cv::Vec3f symmetric_color = initial_ptr[symmetric_index];
         symmetry_ptr[i] = (
            initial_color * initial_color_weight + symmetric_color * symmetric_color_weight
         ) / (initial_color_weight + symmetric_color_weight);
      }
   }
}

void HeadAnalysis::calculateVisibility(const cv::Mat& reference_number_of_query) const
{
   double normalizer;
   minMaxLoc( reference_number_of_query, nullptr, &normalizer, nullptr, nullptr );
   normalizer = 1.0 / normalizer;

   cv::Mat visibility(reference_number_of_query.size(), CV_32FC1);
   for (int j = 0; j < reference_number_of_query.rows; ++j) {
      const auto* reference_ptr = reference_number_of_query.ptr<float>(j);
      auto* visibility_ptr = visibility.ptr<float>(j);
      for (int i = 0; i < reference_number_of_query.cols; ++i) {
         visibility_ptr[i] = 1.0f - 1.0f / exp( 0.5f + reference_ptr[i] * normalizer );
      }
   }

   // the smaller is visibility, the higher are scores. 
   visibility.convertTo( visibility, CV_8UC1, 255.0 );
   imshow( "Visibility", visibility );
}

void HeadAnalysis::frontalize(cv::Mat& frontalized, const cv::Mat& query)
{	
   cv::Matx33f rotation;
   cv::Matx34f projection;
   estimateCameraPose( rotation, projection, query, Reference3DLandmarks );

   cv::Mat mapper_to_query;
   getMapperToQuery( mapper_to_query, projection, query.size() );

   cv::Mat initial_estimated;
   estimateVirtualFace( initial_estimated, query, mapper_to_query );

   cv::Mat reference_number_of_query;
   calculateReferenceNumberOfQuery( reference_number_of_query, mapper_to_query, query );
      
   cv::Mat occlusion_indicator;
   const OCCLUSION occlusion_type = getIndicatorIfOccluded( occlusion_indicator, reference_number_of_query );

   if (occlusion_type == OCCLUSION::NO) {
      initial_estimated.convertTo( frontalized, CV_8UC3 );
      std::cout << "No need to apply soft-symmetry.\n";
   }
   else {
      cv::Mat soft_symmetry;
      applySoftSymmetry( 
         soft_symmetry, 
         initial_estimated, 
         reference_number_of_query,
         occlusion_indicator
      );
      soft_symmetry.convertTo( frontalized, CV_8UC3 );

      calculateVisibility( reference_number_of_query );
   }
}

void HeadAnalysis::lateralize(cv::Mat& lateralized, const cv::Mat& query, const cv::Vec3f& euler_angle_in_degree)
{
   std::vector<cv::Point3f> model_3d_landmarks;
   get3DModelLandmarks( model_3d_landmarks, euler_angle_in_degree );

   cv::Matx33f rotation;
   cv::Matx34f projection;
   estimateCameraPose( rotation, projection, query, model_3d_landmarks );

   cv::Mat mapper_to_query;
   getMapperToQuery( mapper_to_query, projection, query.size() );

   estimateVirtualFace( lateralized, query, mapper_to_query );
   lateralized.convertTo( lateralized, CV_8UC3 );
}

void HeadAnalysis::drawHeadPosition(
   const cv::Vec3f& euler_angle_in_radian, 
   const cv::Mat& query
) const
{
   const float& x = euler_angle_in_radian(0);
   const float& y = euler_angle_in_radian(1);
   const float& z = euler_angle_in_radian(2);
   const cv::Matx33f rotation_x(1.0f, 0.0f, 0.0f, 0.0f, cos( x ), -sin( x ), 0.0f, sin( x ), cos( x ));
   const cv::Matx33f rotation_y(cos( y ), 0.0f, sin( y ), 0.0f, 1.0f, 0.0f, -sin( y ), 0.0f, cos( y ));
   const cv::Matx33f rotation_z(cos( z ), -sin( z ), 0.0f, sin( z ), cos( z ), 0.0f, 0.0f, 0.0f, 1.0f);
   const cv::Matx33f rotation = rotation_z * rotation_y * rotation_x;

   cv::Mat draw = query.clone();
   const cv::Point2f p0(static_cast<float>(query.cols) * 0.5f, static_cast<float>(query.rows) * 0.5f);
   const float scale_factor = 0.333f * std::max( draw.cols, draw.rows );
   cv::Point2f p1(p0.x + scale_factor * rotation(0, 0), p0.y - scale_factor * rotation(1, 0));
   arrowedLine( draw, p0, p1, cv::Scalar(0, 0, 255), 3, 8, 0, 0.25 );

   p1 = cv::Point2f(p0.x + scale_factor * rotation(0, 1), p0.y - scale_factor * rotation(1, 1));
   arrowedLine( draw, p0, p1, cv::Scalar(0, 255, 0), 3, 8, 0, 0.25 );

   p1 = cv::Point2f(p0.x + scale_factor * rotation(0, 2), p0.y - scale_factor * rotation(1, 2));
   arrowedLine( draw, p0, p1, cv::Scalar(255, 0, 0), 3, 8, 0, 0.25 );
   imshow( "Head Position", draw );
}

void HeadAnalysis::estimateHeadPosition(cv::Matx31f& euler_angle_in_degree, const cv::Mat& query)
{
   cv::Matx33f rotation;
   cv::Matx34f projection;
   estimateCameraPose( rotation, projection, query, Reference3DLandmarks );

   const cv::Matx33f to_opencv_system(1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f);
   rotation = to_opencv_system * rotation; 

   cv::Vec3f euler_angle_in_radian;
   euler_angle_in_radian(1) = atan(
      -rotation(2, 0) /
      sqrt( rotation(0, 0) * rotation(0, 0) + rotation(1, 0) * rotation(1, 0) )
   );

   if (cos( euler_angle_in_radian(1) ) < 1e-7f) {
      euler_angle_in_radian(0) = atan( -rotation(1, 2) / rotation(1, 1) );
      euler_angle_in_radian(2) = 0.0f;
   }
   else {
      euler_angle_in_radian(0) = atan( rotation(2, 1) / rotation(2, 2) );
      euler_angle_in_radian(2) = atan( rotation(1, 0) / rotation(0, 0) );
   }
   
   euler_angle_in_degree(0) = euler_angle_in_radian(0) * 180.0f / CV_PI;
   euler_angle_in_degree(1) = euler_angle_in_radian(1) * 180.0f / CV_PI;
   euler_angle_in_degree(2) = euler_angle_in_radian(2) * 180.0f / CV_PI;

   drawHeadPosition( euler_angle_in_radian, query );
}