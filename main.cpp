#include "HeadAnalysis.h"

bool isImageFile(const std::filesystem::path& file_path)
{
   return
      file_path.extension() == ".png" || file_path.extension() == ".PNG" ||
      file_path.extension() == ".jpg" || file_path.extension() == ".JPG";
}

void getAllImagesInFolder(std::vector<std::filesystem::path>& image_paths, const std::filesystem::path& directory_path)
{
   if (!exists( directory_path )) return;

   for (const auto& file : std::filesystem::recursive_directory_iterator(directory_path)) {
      if (isImageFile( file.path() )) {
         image_paths.push_back( file.path() );
      }
   }
}

void frontalization_test()
{
   std::vector<std::filesystem::path> image_set;
   const std::string face_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples/faces";
   getAllImagesInFolder( image_set, face_directory_path + "/lateral" );

   HeadAnalysis head_analyzer;
   for (const auto& file : image_set) {
      const cv::Mat query = cv::imread( file.string() );

      cv::Mat frontalized;
      head_analyzer.frontalize( frontalized, query );
      imshow( "frontalized", frontalized );
      imshow( "query", query );
      cv::waitKey();
   }
}

void lateralization_test()
{
   std::vector<std::filesystem::path> image_set;
   const std::string face_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples/faces";
   getAllImagesInFolder( image_set, face_directory_path + "/frontal" );

   HeadAnalysis head_analyzer;
   for (const auto& file : image_set) {
      const cv::Mat query = cv::imread( file.string() );

      cv::Mat lateralized;
      cv::Vec3f euler_angle(0.0f, 15.0f, 0.0f);
      head_analyzer.lateralize( lateralized, query, euler_angle );
      imshow( "lateralized", lateralized );
      imshow( "query", query );
      cv::waitKey();
   }
}

void head_position_test()
{
   std::vector<std::filesystem::path> image_set;
   const std::string face_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples/faces";
   getAllImagesInFolder( image_set, face_directory_path + "/lateral" );

   HeadAnalysis head_analyzer;
   for (const auto& file : image_set) {
      const cv::Mat query = cv::imread( file.string() );

      cv::Matx31f euler_angle;
      head_analyzer.estimateHeadPosition( euler_angle, query );     
      imshow( "query", query );
      cv::waitKey();
   }
}

int main()
{
   //frontalization_test();
   //lateralization_test();
   head_position_test();
   return 0;
}