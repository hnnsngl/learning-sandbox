#include "loadweights.hpp"

#include <fstream>

#include <opencv2/core.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>

std::vector<std::vector<std::vector<double> > > 
loadWeights(std::string filename)
{
	using namespace boost::spirit;
	using std::vector;
	using MatRow = vector<double>;
	using Matrix = vector<MatRow>;
	std::vector<Matrix> weights;
	
	std::ifstream input(filename);
	if (not input.good()){
		return weights;
	}
	
	std::string str((std::istreambuf_iterator<char>(input)),
	                 std::istreambuf_iterator<char>());

	auto first = str.cbegin();
	                   
	bool status = qi::phrase_parse(first, str.cend(), *( '[' >> (double_ % ',' % ';') >> ']' ),
	                               ascii::space, weights);
	if (not status or (first != str.end())) {
		std::cerr << "  Failed to parse" << std::endl;
		return weights;
	}

	return weights;
}

std::vector<cv::Mat> loadWeightsMat(std::string filename)
{
	using Matrix = std::vector<std::vector<double> >;
	using Weights = std::vector<Matrix>;

	Weights weights = loadWeights(filename);
	std::vector<cv::Mat> results;
	
	for (int i=0; i<weights.size(); i++) {
		const Matrix& matrix = weights[i];
		
		const int nrows = matrix.size(); assert(nrows > 0);
		const int ncols = matrix[0].size(); assert(ncols > 0);

		cv::Mat mat = cv::Mat::zeros(nrows, ncols, CV_64F);
		            
		for (int row = 0; row < nrows; row++) {
			assert(matrix[row].size() == ncols);
			std::copy(matrix[row].begin(), matrix[row].end(), mat.row(row).begin<double>());
		}

		results.push_back(mat);
	}
	
	return results;
}
