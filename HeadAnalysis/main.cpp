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

#define FRONTALIZATION_TEST
#define LATERALIZATION_TEST
#define HEAD_POSITION_TEST
int main()
{
	std::vector<path> image_set;
#if defined(FRONTALIZATION_TEST) || defined(HEAD_POSITION_TEST)
	getAllImagesInFolder( image_set, "FaceSamples/lateral" );
#else
	getAllImagesInFolder( image_set, "FaceSamples/frontal" );
#endif

	HeadAnalysis head_analyzer;
	for (const auto& file : image_set) {
		const Mat query = imread( file.string() );

#if defined FRONTALIZATION_TEST
		Mat frontalized;
		head_analyzer.frontalize( frontalized, query );
		imshow( "frontalized", frontalized );
#elif defined LATERALIZATION_TEST
		Mat lateralized;
		Vec3f euler_angle(0.0f, 15.0f, 0.0f);
		head_analyzer.lateralize( lateralized, query, euler_angle );
		imshow( "lateralized", lateralized );
#elif defined HEAD_POSITION_TEST
		Matx31f euler_angle;
		head_analyzer.estimateHeadPosition( euler_angle, query );
#endif
		
		imshow( "query", query );
		waitKey();
	}

	return 0;
}