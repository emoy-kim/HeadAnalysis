#include "HeadAnalysis.h"

HeadAnalysis::HeadAnalysis() : ScreenWidth( 500 ), ScreenHeight( 500 )
{
	FaceDetector = get_frontal_face_detector();
	deserialize( "../../../Libraries/dlib1916/shape_predictor_68_face_landmarks.dat" ) >> LandmarkDetector;

	get3DModelLandmarks( Reference3DLandmarks, Vec3f(0.0f, 0.0f, 0.0f) );
}

void HeadAnalysis::get3DModelLandmarks(std::vector<Point3f>& model_3d_landmarks, const Vec3f& euler_angle_in_degree)
{
	Mat model_face;
	ModelRenderer.getModelImage( model_face, ModelDepth, 500, 500, 500.0f, euler_angle_in_degree );
	ModelRenderer.getClipToCameraMatrix( ClipToCameraMatrix );

	std::vector<Point2f> model_landmarks;
	detectFaceLandmarks( model_landmarks, model_face );

	model_3d_landmarks.clear();
	for (const auto& landmark : model_landmarks) {
		Point3f unprojected;
		get3DCoordinates( unprojected, landmark );
		model_3d_landmarks.emplace_back( unprojected );
	}
}

bool HeadAnalysis::get3DCoordinates(Point3f& unprojected_3d, const Point2f& projected_2d)
{
	const Point screen(static_cast<int>(round( projected_2d.x )), static_cast<int>(round( projected_2d.y )));
	const auto depth = ModelDepth.at<float>(screen.y, screen.x);
	if (depth == DEPTH_INFINITY) return false;
	
	const Point2f screen_in_opencv_system(projected_2d.x, static_cast<float>(ModelDepth.rows) - projected_2d.y);
	const Matx41f ndc_point(
		2.0f * screen_in_opencv_system.x / static_cast<float>(ModelDepth.cols) - 1.0f,
		(2.0f * screen_in_opencv_system.y / static_cast<float>(ModelDepth.rows) - 1.0f), 
		2.0f * depth - 1.0f,
		1.0f
	);
	
	const Matx41f camera_point = ClipToCameraMatrix * ndc_point;
	unprojected_3d.x = camera_point(0) / camera_point(3);
	unprojected_3d.y = camera_point(1) / camera_point(3);
	unprojected_3d.z = camera_point(2) / camera_point(3);
	return true;
}

bool HeadAnalysis::isValidLandmark(const int& landmark_index) const
{
	// jaw line excluded
	return landmark_index >= 17;
}

bool HeadAnalysis::detectFaceLandmarks(std::vector<Point2f>& landmarks, const Mat& image)
{
	array2d<rgb_pixel> dlib_image;
	assign_image( dlib_image, cv_image<bgr_pixel>(image) );
	std::vector<dlib::rectangle> faces = FaceDetector(dlib_image);
	if (faces.empty()) {
		cout << ">> The image must contain at least one person's face!" << endl;
		return false;
	}

	// only use the first detected person
	full_object_detection estimated_landmarks = LandmarkDetector(dlib_image, faces[0]);
	
	landmarks.clear();
	for (uint i = 0; i < estimated_landmarks.num_parts(); ++i) {
		if (!isValidLandmark( i )) continue;
		landmarks.emplace_back( estimated_landmarks.part(i).x(), estimated_landmarks.part(i).y() );
	}
	return true;
}

void HeadAnalysis::estimateCameraPose(
	Matx33f& rotation, 
	Matx34f& projection, 
	const Mat& query, 
	const std::vector<Point3f>& landmarks_3d
)
{
	std::vector<Point2f> landmarks;
	detectFaceLandmarks( landmarks, query );

	const auto focal_length = static_cast<float>(std::max( query.rows, query.cols ));
	const Matx33f intrinsic(
		focal_length, 0.0f, static_cast<float>(query.cols) * 0.5f,
		0.0f, focal_length, static_cast<float>(query.cols) * 0.5f,
		0.0f, 0.0f, 1.0f
	);

	Mat rotation_vector(3, 1, CV_32FC1);
	Mat translation_vector(3, 1, CV_32FC1);
	solvePnP(
		landmarks_3d, 
		landmarks, 
		Mat(intrinsic), 
		Mat(), 
		rotation_vector, 
		translation_vector, 
		SOLVEPNP_ITERATIVE
	);
	Matx31f translation = translation_vector;
	Mat rotation_matrix;
	Rodrigues( rotation_vector, rotation_matrix );
	rotation = rotation_matrix;

	projection = intrinsic * Matx34f(
		rotation(0, 0), rotation(0, 1), rotation(0, 2), translation(0),
		rotation(1, 0), rotation(1, 1), rotation(1, 2), translation(1),
		rotation(2, 0), rotation(2, 1), rotation(2, 2), translation(2)
	);
}

void HeadAnalysis::getMapperToQuery(Mat& mapper_to_query, const Matx34f& projection, const Size& query_size)
{
	mapper_to_query.create( ScreenHeight, ScreenWidth, CV_32FC2 );

	const auto width_boundary = static_cast<float>(query_size.width - 1);
	const auto height_boundary = static_cast<float>(query_size.height - 1);
	for (int j = 0; j < mapper_to_query.rows; ++j) {
		auto* mapper_ptr = mapper_to_query.ptr<Vec2f>(j);
		for (int i = 0; i < mapper_to_query.cols; ++i) {
			Point3f unprojected;
			Point2f screen_point(static_cast<float>(i), static_cast<float>(j));
			if (get3DCoordinates( unprojected, screen_point )) {
				Matx31f query_point = projection * Matx41f(unprojected.x, unprojected.y, unprojected.z, 1.0f);
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
	Vec3f& bgr_color, 
	const Mat& image, 
	const Vec2f& point
) const
{
	const int x0 = static_cast<int>(floor( point(0) ));
	const int y0 = static_cast<int>(floor( point(1) ));
	const int x1 = std::min( x0 + 1, image.cols - 1 );
	const int y1 = std::min( y0 + 1, image.rows - 1 );
	const auto tx = point(0) - static_cast<float>(x0);
	const auto ty = point(1) - static_cast<float>(y0);

	const auto* curr_row = image.ptr<Vec3b>(y0);
	const auto* next_row = image.ptr<Vec3b>(y1);
	bgr_color = 
		static_cast<Vec3f>(curr_row[x0]) * (1.0f - tx) * (1.0f - ty) + 
		static_cast<Vec3f>(curr_row[x1]) * tx * (1.0f - ty) + 
		static_cast<Vec3f>(next_row[x0]) * (1.0f - tx) * ty + 
		static_cast<Vec3f>(next_row[x1]) * tx * ty;
}

void HeadAnalysis::estimateVirtualFace(Mat& virtualized, const Mat& query, const Mat& mapper_to_query) const
{
	virtualized.create( ScreenHeight, ScreenWidth, CV_32FC3 );
	for (int j = 0; j < virtualized.rows; ++j) {
		const auto* mapper_ptr = mapper_to_query.ptr<Vec2f>(j);
		auto* estimated_ptr = virtualized.ptr<Vec3f>(j);
		for (int i = 0; i < virtualized.cols; ++i) {
			if (mapper_ptr[i](0) < 0.0f) {
				estimated_ptr[i] = Vec3f(255.0f, 255.0f, 255.0f);
			}
			else {
				Vec3f bgr_color;
				getBilinearInterpolatedColor( bgr_color, query, mapper_ptr[i] );
				estimated_ptr[i] = bgr_color;
			}
		}
	}
}

void HeadAnalysis::calculateReferenceNumberOfQuery(
	Mat& reference_number_of_query, 
	const Mat& mapper_to_query, 
	const Mat& query
) const
{
	Mat number_of_being_referenced = Mat::zeros(query.size(), CV_32SC1);
	for (int j = 0; j < mapper_to_query.rows; ++j) {
		const auto* mapper_ptr = mapper_to_query.ptr<Vec2f>(j);
		for (int i = 0; i < mapper_to_query.cols; ++i) {
			if (mapper_ptr[i](0) >= 0.0f) {
				const int x = static_cast<int>(round( mapper_ptr[i](0) ));
				const int y = static_cast<int>(round( mapper_ptr[i](1) ));
				const int data_index = y * query.cols + x;
				reinterpret_cast<int*>(number_of_being_referenced.data)[data_index]++;
			}
		}
	}

	reference_number_of_query = Mat::zeros(mapper_to_query.size(), CV_32SC1);
	for (int j = 0; j < mapper_to_query.rows; ++j) {
		const auto* mapper_ptr = mapper_to_query.ptr<Vec2f>(j);
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
		Size(15, 15), 30.0, BORDER_REPLICATE
	);
}

int HeadAnalysis::getIndicatorIfOccluded(Mat& occlusion_indicator, const Mat& reference_number_of_query) const
{
	const Mat& reference = reference_number_of_query;
	const auto total_sum = round( sum( reference )[0] );
	const int middle = reference.cols / 2;
	const Rect left_part(0, 0, middle, reference.rows);
	const Rect right_part(middle, 0, reference.cols - middle, reference.rows);
	const auto left_sum = static_cast<int>(round( sum( reference(left_part) )[0] ));
	const auto right_sum = static_cast<int>(round( sum( reference(right_part) )[0] ));
	const int sum_difference = left_sum - right_sum;

	if (fabs( sum_difference ) * 4.5 < total_sum) return NO_OCCLUSION;

	const int occlusion_position = sum_difference > 0 ? LEFT_OCCLUSION : RIGHT_OCCLUSION;
	occlusion_indicator = Mat::ones(reference.size(), CV_32FC1);
	switch (occlusion_position) {
		case LEFT_OCCLUSION:
			occlusion_indicator(left_part) = 0.0f;
			break;
		case RIGHT_OCCLUSION:
			occlusion_indicator(right_part) = 0.0f;
			break;
		default: break;
	}
	GaussianBlur(
		occlusion_indicator, 
		occlusion_indicator, 
		Size(15, 15), 60.0, BORDER_REPLICATE
	);
	return occlusion_position;
}

void HeadAnalysis::applySoftSymmetry(
	Mat& soft_symmetry, 
	const Mat& initial_estimated, 
	const Mat& reference_number_of_query,
	const Mat& occlusion_indicator
) const
{
	double normalizer;
	minMaxLoc( reference_number_of_query, nullptr, &normalizer, nullptr, nullptr );
	normalizer = 1.0 / normalizer;

	soft_symmetry.create( initial_estimated.size(), initial_estimated.type() );
	for (int j = 0; j < soft_symmetry.rows; ++j) {
		const auto* initial_ptr = initial_estimated.ptr<Vec3f>(j);
		const auto* reference_ptr = reference_number_of_query.ptr<float>(j);
		const auto* indicator_ptr = occlusion_indicator.ptr<float>(j);
		auto* symmetry_ptr = soft_symmetry.ptr<Vec3f>(j);
		
		for (int i = 0; i < soft_symmetry.cols; ++i) {
			const int symmetric_index = soft_symmetry.cols - 1 - i;
			const float visibility_factor = 1.0f / exp( 0.5f + reference_ptr[i] * static_cast<float>(normalizer) );
			const float initial_color_weight = indicator_ptr[i] + visibility_factor * indicator_ptr[symmetric_index];
			const float symmetric_color_weight = (1.0f - visibility_factor) * indicator_ptr[symmetric_index];

			const Vec3f initial_color = initial_ptr[i];
			const Vec3f symmetric_color = initial_ptr[symmetric_index];
			symmetry_ptr[i] = (
				initial_color * initial_color_weight + symmetric_color * symmetric_color_weight
			) / (initial_color_weight + symmetric_color_weight);
		}
	}
}

void HeadAnalysis::calculateVisibility(const Mat& reference_number_of_query) const
{
	double normalizer;
	minMaxLoc( reference_number_of_query, nullptr, &normalizer, nullptr, nullptr );
	normalizer = 1.0 / normalizer;

	Mat visibility(reference_number_of_query.size(), CV_32FC1);
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

void HeadAnalysis::frontalize(Mat& frontalized, const Mat& query)
{	
	Matx33f rotation;
	Matx34f projection;
	estimateCameraPose( rotation, projection, query, Reference3DLandmarks );

	Mat mapper_to_query;
	getMapperToQuery( mapper_to_query, projection, query.size() );

	Mat initial_estimated;
	estimateVirtualFace( initial_estimated, query, mapper_to_query );

	Mat reference_number_of_query;
	calculateReferenceNumberOfQuery( reference_number_of_query, mapper_to_query, query );
		
	Mat occlusion_indicator;
	const int occlusion_type = getIndicatorIfOccluded( occlusion_indicator, reference_number_of_query );

	if (occlusion_type == NO_OCCLUSION) {
		initial_estimated.convertTo( frontalized, CV_8UC3 );
		cout << "No need to apply soft-symmetry." << endl;
	}
	else {
		Mat soft_symmetry;
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

void HeadAnalysis::lateralize(Mat& lateralized, const Mat& query, const Vec3f& euler_angle_in_degree)
{
	std::vector<Point3f> model_3d_landmarks;
	get3DModelLandmarks( model_3d_landmarks, euler_angle_in_degree );

	Matx33f rotation;
	Matx34f projection;
	estimateCameraPose( rotation, projection, query, model_3d_landmarks );

	Mat mapper_to_query;
	getMapperToQuery( mapper_to_query, projection, query.size() );

	estimateVirtualFace( lateralized, query, mapper_to_query );
	lateralized.convertTo( lateralized, CV_8UC3 );
}

void HeadAnalysis::drawHeadPosition(
	const Vec3f& euler_angle_in_radian, 
	const Mat& query
) const
{
	const float& x = euler_angle_in_radian(0);
	const float& y = euler_angle_in_radian(1);
	const float& z = euler_angle_in_radian(2);
	const Matx33f rotation_x(1.0f, 0.0f, 0.0f, 0.0f, cos( x ), -sin( x ), 0.0f, sin( x ), cos( x ));
	const Matx33f rotation_y(cos( y ), 0.0f, sin( y ), 0.0f, 1.0f, 0.0f, -sin( y ), 0.0f, cos( y ));
	const Matx33f rotation_z(cos( z ), -sin( z ), 0.0f, sin( z ), cos( z ), 0.0f, 0.0f, 0.0f, 1.0f);
	const Matx33f rotation = rotation_z * rotation_y * rotation_x;

	Mat draw = query.clone();
	const Point2f p0(static_cast<float>(query.cols) * 0.5f, static_cast<float>(query.rows) * 0.5f);
	const float scale_factor = 0.333f * std::max( draw.cols, draw.rows );
	Point2f p1(p0.x + scale_factor * rotation(0, 0), p0.y - scale_factor * rotation(1, 0));
	arrowedLine( draw, p0, p1, Scalar(0, 0, 255), 3, 8, 0, 0.25 );

	p1 = Point2f(p0.x + scale_factor * rotation(0, 1), p0.y - scale_factor * rotation(1, 1));
	arrowedLine( draw, p0, p1, Scalar(0, 255, 0), 3, 8, 0, 0.25 );

	p1 = Point2f(p0.x + scale_factor * rotation(0, 2), p0.y - scale_factor * rotation(1, 2));
	arrowedLine( draw, p0, p1, Scalar(255, 0, 0), 3, 8, 0, 0.25 );
	imshow( "Head Position", draw );
}

void HeadAnalysis::estimateHeadPosition(Matx31f& euler_angle_in_degree, const Mat& query)
{
	Matx33f rotation;
	Matx34f projection;
	estimateCameraPose( rotation, projection, query, Reference3DLandmarks );

	const Matx33f to_opencv_system(1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f);
	rotation = to_opencv_system * rotation; 

	Vec3f euler_angle_in_radian;
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