#pragma once

#include <vector>
#include <string>

struct Options
{
	bool parse(int argc, char **);

	int count = 0; // 0 -- use all training data
	int loops = 0; // 0 -- no iterations
	int batch = 0; // 0 -- full batch

	double lambda = 1.0;
	double alpha = 1.0;
	double epsilon = 1.0;

	std::string prefix = "nn";
	std::string basename;
	std::vector<int> layers = {};
};
