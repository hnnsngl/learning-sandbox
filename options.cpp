#include "options.hpp"

#include <iostream>

#include <boost/program_options.hpp>

bool Options::parse(int argc, char **argv)
{
	namespace po = boost::program_options;

	po::options_description optionsDescription("Options");
	// clang-format off
	optionsDescription.add_options()("help,h", "print help message")(
	    "layers", po::value<std::vector<int>>(&layers), "hidden layer sizes (default: none)")(
	    "batch,b", po::value<int>(&batch), "batch size (default: full batch)")(
	    "iterations,n", po::value<int>(&loops), "number of full iterations")(
	    "count,c", po::value<int>(&count), "restrict training set to size")(
	    "lambda", po::value<double>(&lambda), "L2 regularization parameter")(
	    "alpha", po::value<double>(&alpha), "gradient descent step factor")(
	    "epsilon", po::value<double>(&epsilon), "bound for random weights initialization")(
	    "prefix", po::value<std::string>(&prefix), "prefix for output files");
	// clang-format on

	po::positional_options_description positionalOptions;
	positionalOptions.add("layers", -1);

	po::variables_map options;
	try {
		po::store(po::command_line_parser(argc, argv)
		              .options(optionsDescription)
		              .positional(positionalOptions)
		              .run(),
		          options);
		po::notify(options);
	} catch (po::unknown_option) {
		std::cerr << optionsDescription << std::endl;
		return false;
	}

	if (options.count("help")) {
		std::cerr << optionsDescription << std::endl;
		return false;
	}

	// construct file name base
	std::stringstream base;
	if (layers.size() > 0)
		base << prefix << "-" << layers[0];
	for (int l = 1; l < layers.size(); l++)
		base << "x" << layers[l];
	basename = base.str();

	return true;
}
