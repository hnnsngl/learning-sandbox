#pragma once

#include <vector>
#include <utility>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// display montage of images with classification labels for a dataset (to show false
// classifications)
template <class Dataset>
cv::Mat montageList(const Dataset &dataset, const std::vector<std::pair<int, int>> &list,
                    int tiles_per_row = 30)
{
	using cv::Mat;

	const int count = list.size();
	const int xMargin = 2;
	const int yMargin = 14;

	const int width = dataset.imgCols + xMargin;
	const int height = dataset.imgRows + yMargin;

	assert(dataset.images.size() > 0);
	const int type = dataset.images[0].type();

	if (list.size() == 0)
		return Mat(0, 0, type);

	Mat mat = Mat::ones((count / tiles_per_row + 1) * height, tiles_per_row * width, type);

	for (int i = 0; i < list.size(); i++) {
		int x = (i % tiles_per_row) * width;
		int y = (i / tiles_per_row) * height;
		int id = list[i].first;
		dataset.images[id].copyTo(mat(cv::Rect(x, y, dataset.imgCols, dataset.imgRows)));

		std::string label =
		    std::to_string(list[i].second) + "/" + std::to_string(dataset.labels[id]);

		cv::putText(mat, label, cv::Point(x, y + height - 2), cv::FONT_HERSHEY_SIMPLEX, 0.4,
		            cv::Scalar({0.0, 0.0, 0.0, 0.0}));
	}
	return mat;
}
