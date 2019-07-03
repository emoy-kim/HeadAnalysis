#include "HeadAnalysis.h"
#include <filesystem>

using namespace std::experimental::filesystem;

bool isImageFile(const path& file_path)
{
   return
      file_path.extension() == ".png" || file_path.extension() == ".PNG" ||
      file_path.extension() == ".jpg" || file_path.extension() == ".JPG";
}

void getAllImagesInFolder(std::vector<path>& image_paths, const path& directory_path)
{
   if (!exists( directory_path )) return;

   for (const auto& file : recursive_directory_iterator(directory_path)) {
      if (isImageFile( file.path() )) {
         image_paths.push_back( file.path() );
      }
   }
}

void frontalization_test()
{
   std::vector<path> image_set;
   getAllImagesInFolder( image_set, "FaceSamples/lateral" );

   HeadAnalysis head_analyzer;
   for (const auto& file : image_set) {
      const Mat query = imread( file.string() );

      Mat frontalized;
      head_analyzer.frontalize( frontalized, query );
      imshow( "frontalized", frontalized );
      imshow( "query", query );
      waitKey();
   }
}

void lateralization_test()
{
   std::vector<path> image_set;
   getAllImagesInFolder( image_set, "FaceSamples/frontal" );

   HeadAnalysis head_analyzer;
   for (const auto& file : image_set) {
      const Mat query = imread( file.string() );

      Mat lateralized;
      Vec3f euler_angle(0.0f, 15.0f, 0.0f);
      head_analyzer.lateralize( lateralized, query, euler_angle );
      imshow( "lateralized", lateralized );
      imshow( "query", query );
      waitKey();
   }
}

void head_position_test()
{
   std::vector<path> image_set;
   getAllImagesInFolder( image_set, "FaceSamples/lateral" );

   HeadAnalysis head_analyzer;
   for (const auto& file : image_set) {
      const Mat query = imread( file.string() );

      Matx31f euler_angle;
      head_analyzer.estimateHeadPosition( euler_angle, query );     
      imshow( "query", query );
      waitKey();
   }
}

int main()
{
   //frontalization_test();
   lateralization_test();
   //head_position_test();
   return 0;
}