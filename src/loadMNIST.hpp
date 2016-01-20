#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// basic container to get started using the MNIST dataset

struct DatasetMNIST
{
	const int32_t magic_MNISTlabels = 0x00000801;
	const int32_t magic_MNISTimages = 0x00000803;

	typedef std::vector<uint8_t> Labels;
	typedef std::vector<uint8_t> Pixels;
	typedef std::vector<Pixels> Images;

      public:
	// construct from MNIST data files
	DatasetMNIST(std::string filenameImages, std::string filenameLabels);

	std::vector<uint8_t> labels;
	std::vector<cv::Mat> images;
	std::vector<std::string> names;

	int imgRows;
	int imgCols;
	int imgSize;
	const int imgDepth = 1;
	int size = 0;

	bool loadLabels(std::string filename);
	bool loadImages(std::string filename);

	void showImage(int i) const;

	// helper to read int32 in Big Endian
	uint32_t readuint32BE(std::ifstream &ifs) const;
};

DatasetMNIST::DatasetMNIST(std::string filenameImages, std::string filenameLabels)
    : names({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
{
	if (not loadLabels(filenameLabels))
		std::cerr << "Failed to load Labels" << std::endl;

	if (not loadImages(filenameImages))
		std::cerr << "Failed to load Images" << std::endl;

	if (labels.size() != images.size())
		std::cerr << "WARNING: labels (" << labels.size() << ") and images ("
		          << images.size() << ") counts differ" << std::endl;
}

uint32_t DatasetMNIST::readuint32BE(std::ifstream &ifs) const
{
	uint8_t bytes[4];
	ifs.read(reinterpret_cast<char *>(bytes), 4);

	return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

bool DatasetMNIST::loadLabels(std::string filename)
{
	std::ifstream ifs(filename, std::ios::binary);
	ifs.seekg(0);

	uint32_t magic = readuint32BE(ifs);
	uint32_t count = readuint32BE(ifs);

	if (not ifs.good()) {
		std::cerr << "(loadLabels) ERROR cannot open file: " << filename << std::endl;
		return false;
	}

	if (magic != magic_MNISTlabels) {
		std::cerr << "(loadLabels) ERROR invalid file magic (" << magic << "): " << filename
		          << std::endl;
		return false;
	}

	std::cerr << "(loadLabels) INFO file " << filename << " contains " << count << " labels"
	          << std::endl;

	labels.clear();
	labels.resize(count);

	ifs.read(reinterpret_cast<char *>(labels.data()), count);

	return true;
}

bool DatasetMNIST::loadImages(std::string filename)
{
	std::ifstream ifs(filename, std::ios::binary);
	ifs.seekg(0);

	if (not ifs.good()) {
		std::cerr << "(loadImages) ERROR cannot open file: " << filename << std::endl;
		return false;
	}

	int32_t magic = readuint32BE(ifs);
	int32_t count = readuint32BE(ifs);
	int32_t nrows = readuint32BE(ifs);
	int32_t ncols = readuint32BE(ifs);

	imgRows = nrows;
	imgCols = ncols;
	imgSize = nrows * ncols;

	if (magic != magic_MNISTimages) {
		std::cerr << "(loadImages ERROR invalid file magic (" << magic << "): " << filename
		          << std::endl;
		return false;
	}

	if ((nrows == 0) or (ncols == 0)) {
		std::cerr << "(loadImages ERROR number of rows x cols (" << nrows << " x " << ncols
		          << ") is zero" << std::endl;
		return false;
	}

	std::cerr << "(loadImages) INFO file " << filename << " contains " << count << " (" << nrows
	          << " x " << ncols << ") images" << std::endl;

	images.clear();
	images.reserve(count);

	char buffer[imgSize];
	for (int32_t i = 0; i < count; i++) {
		Pixels pixels(nrows * ncols, 0);
		ifs.read(buffer, imgSize);
		cv::Mat img(cv::Size(imgRows, imgCols), CV_8U, buffer, cv::Mat::AUTO_STEP);
		img.convertTo(img, CV_64F);
		img = img / 255.0;
		images.push_back(img);
	}

	size = images.size();

	std::cerr << "loaded " << images.size() << " images " << cv::Size(imgRows, imgCols)
	          << std::endl;

	return true;
}

void DatasetMNIST::showImage(int i) const
{
	int label = labels[i];
	std::string name = (names.size() > label) ? " " + names[i] : "";
	std::cerr << label << name << std::endl;

	cv::imshow("Display window", images[i].reshape(1, imgRows));
}
