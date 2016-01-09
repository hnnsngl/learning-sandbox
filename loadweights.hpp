#pragma once

#include <vector>
#include <string>

#include <opencv2/core.hpp>

std::vector<std::vector<std::vector<double> > > 
loadWeights(std::string filename);

std::vector<cv::Mat> loadWeightsMat(std::string filename);

