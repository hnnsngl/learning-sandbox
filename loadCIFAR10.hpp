#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using cv::Mat;

// basic container to get started using binary CIFAR dataset

struct DatasetCIFAR10
{
	const int imgRows = 32;
	const int imgCols = 32;
	const int imgDepth = 3;
	const int imgSize = imgRows * imgCols;

	int size = 0;

	// construct from CIFAR10 data files
	DatasetCIFAR10(std::string batchfile, std::string labelfile = "")
	{
		loadBatch(batchfile);
		if (labelfile != "")
			loadNames(labelfile);
	}
	DatasetCIFAR10(std::initializer_list<std::string> batchfiles, std::string labelfile = "")
	{
		for (auto &filename : batchfiles)
			loadBatch(filename);
		if (labelfile != "")
			loadNames(labelfile);
	}

	std::vector<uint8_t> labels;
	std::vector<Mat> images;
	std::vector<std::string> names;
	
	bool loadBatch(std::string filename);
	bool loadNames(std::string filename);

	void showBatch(int delay = 200) const;
	void showImage(int i) const;
	void clear();
};

void DatasetCIFAR10::clear(){
	labels.clear();
	images.clear();
}

bool DatasetCIFAR10::loadBatch(std::string filename)
{
	std::ifstream ifs(filename);

	if (not ifs.good()) {
		std::cerr << "(loadImages) ERROR cannot open file: " << filename << std::endl;
		return false;
	}

	while (not ifs.eof()) {
		char lbl = 0;
		ifs.read(&lbl, 1);
		char buffer[imgSize * 3];
		ifs.read(buffer, imgSize*3);
		if (ifs.eof()) break;
		// load channels, convert to float, normalize
		cv::Size size(imgRows,imgCols);
		std::vector<Mat> mats( {Mat(size, CV_8UC1, buffer, Mat::AUTO_STEP),
					Mat(size, CV_8UC1, buffer+imgSize, Mat::AUTO_STEP),
					Mat(size, CV_8UC1, buffer+2*imgSize, Mat::AUTO_STEP)} );
		Mat img(size, CV_8UC3);
		cv::merge(mats, img);
		img.convertTo(img, CV_64FC3);
		img = img / 255.0f;
		img = img.reshape(1, imgRows);
		
		labels.push_back(lbl);
		images.push_back(img);
	}
	size = images.size();
	return true;
}

bool DatasetCIFAR10::loadNames(std::string filename)
{
	std::ifstream ifs(filename);
	if (not ifs.good()) return false;
	
	while (ifs.good()) {
		std::string line;
		std::getline(ifs, line);
		if (ifs.good()) names.push_back(line);
	}
	return true;
}

void DatasetCIFAR10::showImage(int i) const
{
	int label = labels[i];
	std::string name = (names.size() > label) ? " " + names[i] : "";
	std::cerr << label << name << std::endl;

	cv::imshow("Display window", images[i].reshape(3, imgRows));
}

void DatasetCIFAR10::showBatch(int delay) const
{
	for (int i=0; i<images.size(); i++) {
		showImage(i);
		cv::waitKey(delay);
	}
}
