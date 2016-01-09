
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "loadMNIST.hpp"
#include "loadweights.hpp"
#include "montagelist.hpp"
#include "options.hpp"

using cv::Mat;

Mat &apply(Mat &dst, const Mat &src, double fun(double));
Mat apply(double fun(double), const Mat &mat);

double sqrsum(const Mat &src);

double sigmoid(double x) { return 1.0f / (1.0f + std::exp(-x)); }
double sigmoidGrad(double x) { return std::exp(-x) * std::pow(1.0f + std::exp(-x), -2.0f); }

Mat log(const Mat &mat) { return apply(&std::log, mat); }

Mat computeInputVector(const Mat &img);
Mat computeActivation(const Mat &input);

// same as computeActivation without adding the bias unit
Mat computeOutputVector(const Mat &input) { return apply(&sigmoid, input); }

int getClassification(const Mat& activation);

std::vector<Mat> makeTeachingVectors(int size);


#if defined DEBUG
constexpr bool debug_output = true;
#else
constexpr bool debug_output = false;
#endif

#define debuglog if(not debug_output) {} else std::cerr

int main(int argc, char **argv)
{
	Options options;
	options.layers = {300, 100};
	options.prefix = "mnist";
	if (not options.parse(argc, argv)) return 1;
	
	// optimization parameters
	double alpha = options.alpha;
	double lambda = options.lambda;
	double epsilon = options.epsilon;
	
	DatasetMNIST trainingSet("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte");

	// mini-batch parameters
	int count = (options.count > 0) ? options.count : trainingSet.size;
	int batch = (options.batch > 0) ? options.batch : trainingSet.size;
	int loops = (options.loops > 0) ? options.loops : 20;

	// fix network architecture: input layer, hidden layer(s),
	// output layer
	const int input_size = trainingSet.imgSize * trainingSet.imgDepth;
	const int output_size = 10;
	std::vector<int> architecture({input_size, output_size});
	architecture.insert(architecture.begin()+1, options.layers.begin(), options.layers.end());

	std::cout << "# alpha = " << alpha << "\n# lambda = " << lambda << "\n# epsilon = " << epsilon
	          << "\n# count = " << count << "\n# batch = " << batch << "\n# loops = " << loops << "\n\n"
	          << "# NN Architecture: " << architecture[0];
	for (int l=1; l<architecture.size(); l++) std::cout << " x " << architecture[l];
	std::cout << "\n";

	// prepare and initialize weight matrices W_i for given
	// architecture, so that
	//
	// Z_(i+1) = W_i) * A_i
	std::vector<Mat> weights = loadWeightsMat(options.basename + "-weights");
	for (int i = 1; i < architecture.size(); i++) {
		cv::Size size = cv::Size(architecture[i - 1] + 1, architecture[i]);
		
		if ((weights.size() >= i) and weights[i-1].size() == size) {
			std::cerr << "Loaded Weight Matrix[" << i << "] " << weights[i-1].t().size() << std::endl;
		} else {
			Mat weight(size, CV_64F);
			cv::randu(weight, cv::Scalar::all(-epsilon), cv::Scalar::all(epsilon));
			weights.push_back(weight);
			std::cerr << "Initial Weight Matrix[" << i << "] " << weight.t().size() << std::endl;
		}
	}

	// create teaching signal vectors
	std::vector<Mat> yk = makeTeachingVectors(output_size);

	std::vector<double> Jseries;

	// do mini-batch gradient descent, 10 times round
	for (int nb = 0; nb < loops * count / batch; nb++) {
		double J = 0;

		// index of the last layer in the weight, input and activation vectors
		const int last = architecture.size() - 1;

		// prepare weight gradients
		std::vector<Mat> grad(last);  // cost gradients for weights
		for (int l = 0; l < last; l++)
			grad[l] = Mat::zeros(weights[l].size(), weights[l].type());

#pragma omp parallel for
		for (int i = nb * batch; i < (nb + 1) * batch; i++) {
			int id = i % count;

			std::vector<Mat> Z(last + 1); // input vectors
			std::vector<Mat> A(last + 1); // activation vectors
			std::vector<Mat> D(last + 1); // error vectors for backpropagation

			// feed forward and check
			for (int l = 0; l < last; l++) {
				if (l == 0)
					A[l] = computeInputVector(trainingSet.images[id]);
				else
					A[l] = computeActivation(Z[l]);
				Z[l + 1] = weights[l] * A[l];
			}
			A[last] = computeOutputVector(Z[last]);

			// delta2 = (Theta2' * delta3)(2:hidden_layer_size+1) .* sigmoidGradient(Z2);
			int label = trainingSet.labels[id];

			D[last] = A[last] - yk[label];
			for (int l = last - 1; l>0; --l) {
				D[l] = (weights[l].t() * D[l+1])(cv::Rect(0, 1, 1, architecture[l]));
				D[l] = D[l].mul(apply(Z[l], Z[l], &sigmoidGrad));
			}
#pragma omp critical
			{
				// J = J + sum(-yk.*log(A3) - (1 - yk).*(log(1 - A3))) / m;
				J +=
				    cv::sum(-1.0f * yk[label].mul(log(A[last])) - (1.0f - yk[label]).mul(log(1.0f - A[last])))[0];
				for (int l=0; l<last; l++) grad[l] += D[l+1] * A[l].t();
			}
		}

		J /= batch;
		
		// add regularization term to gradient:
		// if (lambda != 0) {  R1 = Theta1 .* (lambda / m);
		//     R1 = Theta1 .* (lambda / m);
		//     R1(:,1) = zeros(1,hidden_layer_size);
		//     R2 = Theta2 .* (lambda / m);
		//     R2(:,1) = zeros(1,num_labels);
		//     D1 = D1 + R1;
		//     D2 = D2 + R2;
		// }

		// std::cerr << "\nRegularization";
		std::vector<Mat> R(last);
		double Jreg = 0;
		for (int l=0; l<last; l++) {
			R[l] = weights[l] * (lambda / batch);
			R[0](cv::Rect(0, 0, 1, architecture[l+1])) = Mat::zeros(1, architecture[l+1], CV_64F);
			grad[l] = grad[l] / batch + R[l];

			Jreg += sqrsum(weights[l]);
			weights[l] -= alpha * grad[l];
		}
		Jreg *= lambda / (2 * batch);
		double Jtot = J + Jreg;

		std::cout << nb << "\tJ = " << J << " + " << Jreg << " = " << Jtot << std::endl;

		Jseries.push_back(Jtot);
		
		int wndSize = count / batch;
		double JWndMean = Jtot;
		if (Jseries.size() > wndSize) {
			JWndMean = std::accumulate(Jseries.rbegin(), Jseries.rbegin() + wndSize, 0.0) / wndSize;
		}
		if (Jtot > JWndMean) {
			alpha = alpha * 0.9;
			std::cout << nb << "\tJ = " << Jtot << "\talpha --> " << alpha << "\n";
		}
		if (alpha < 1e-14) {
			break;
		}
	}

	if (loops > 0) {
		// store weights
		std::ofstream osweights(options.basename + "-weights");
		for (int i = 0; i < architecture.size() - 1; i++)
			osweights << weights[i] << std::endl;

		// store cost function values
		std::ofstream oscosts(options.basename + "-costs");
		for (int i = 0; i < Jseries.size(); i++)
			oscosts << i << "\t" << Jseries[i] << "\n";
	}

	// start testing
	DatasetMNIST testset("MNIST/t10k-images-idx3-ubyte", "MNIST/t10k-labels-idx1-ubyte");
	int correct = 0;
	Mat stats = Mat::eye(10, 10, CV_64F);
	std::vector<std::pair<int,int> > errors; // image ids of wrong guesses

	std::vector<Mat> Z(architecture.size());
	std::vector<Mat> A(architecture.size());
	
	for (int id = 0; id < testset.size; id++) {
		int last = architecture.size() - 1; // index of the last layer in the weight, input and activation vectors
		// feed forward and check
		for (int l=0; l < last; l++) {
			if (l == 0)
				A[l] = computeInputVector(testset.images[id]);
			else
				A[l] = computeActivation(Z[l]);
			Z[l+1] = weights[l] * A[l];
		}
		A[last] = computeOutputVector(Z[last]);

		// find classification
		int classification = getClassification(A[last]);
		
		int label = testset.labels[id];
		stats += yk[label] * A[last].t();

		if (classification == label)
			++correct;
		else {
			errors.push_back(std::make_pair(id, classification));
		}
	}

	std::cout << "# Total correct: " << correct << " / " << testset.size << " = "
	          << static_cast<double>(correct) / testset.size << "\n# Total classification error: "
	          << static_cast<double>(testset.size - correct) / testset.size  << std::endl;

	std::ofstream osconfidence(options.basename + "-confidence");
	osconfidence << "# row: label, column: estimate\n" << stats / testset.size << std::endl;

	std::string title = "classification errors";
	cv::namedWindow(title);
	Mat display = montageList(testset, errors);
	cv::imshow(title, display);
	while (cv::waitKey(0) != 27);
}


double sqrsum(const Mat &src)
{
	assert(src.isContinuous());
	const double *x = src.ptr<double>(0);
	cv::Size size = src.size();
	double sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < size.width * size.height; i++) {
		sum += x[i] * x[i];
	}
	return sum;
}

Mat &apply(Mat &dst, const Mat &src, double fun(double))
{
	assert(src.isContinuous());
	assert(dst.isContinuous());
	assert(dst.size() == src.size());

	int n = src.size().width * src.size().height;

	const double *x = src.ptr<double>(0);
	double *y = dst.ptr<double>(0);
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		y[i] = fun(x[i]);
	return dst;
}

Mat apply(double fun(double), const Mat &mat)
{
	Mat result(mat.size(), mat.type());
	return apply(result, mat, fun);
}

Mat computeInputVector(const Mat &img)
{
	Mat input = img.reshape(1, 1).t();
	cv::Size size = input.size();
	size.height++;
	Mat result(size, input.type());

	assert(input.isContinuous());
	assert(result.isContinuous());

	const double *in = input.ptr<double>(0);
	double *out = result.ptr<double>(0);
	out[0] = 1.0f;
	out++;
	for (int i = 0; i < input.size().height; i++)
		out[i] = in[i];
	return result;
}

// compute activation value and add bias term
Mat computeActivation(const Mat &input)
{
	cv::Size size = input.size();
	size.height++;
	Mat result(size, input.type());

	assert(input.isContinuous());
	assert(result.isContinuous());
	const double *zvec = input.ptr<double>(0);
	double *avec = result.ptr<double>(0);
	avec[0] = 1.0f;
	avec++;

#pragma omp parallel for
	for (int i = 0; i < size.height - 1; i++) {
		avec[i] = 1.0 / (1.0 + std::exp(-(zvec[i])));
	}
	return result;
}

int getClassification(const Mat& activation) {
	int idmax[2];
	double max;
	minMaxIdx(activation, 0, &max, 0, idmax);
	return idmax[0];
}

std::vector<Mat> makeTeachingVectors(int size)
{
	std::vector<Mat> yk;
	// use individual vectors
	for (int i = 0; i < size; i++) {
		Mat y = Mat::zeros(size, 1, CV_64F);
		y.ptr<double>(0)[i] = 1.0f;
		yk.push_back(y);
	}

#if defined DEBUG
	for (int k = 0; k < size; k++)
		std::cerr << "k = " << k << " --> yk'" << yk[k].size() << " = " << yk[k].t() << "\t y" << k << "(" << k
		          << ") = " << yk[k].at<double>(0, k) << "\n";
	cv::waitKey(-1);
#endif

	return yk;
}

