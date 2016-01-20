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
std::vector<Mat> makeTeachingVectors(int size);

#if defined DEBUG
constexpr bool debug_output = true;
#else
constexpr bool debug_output = false;
#endif

#define debuglog                                                                                   \
	if (not debug_output) {                                                                    \
	} else                                                                                     \
	std::cerr

int main(int argc, char **argv)
{
	DatasetMNIST trainingSet("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte");

	// optimization parameters
	double alpha = 1.0f;
	double lambda = 1.0f;
	double epsilon = 1.0f;

	// mini-batch parameters
	int count = trainingSet.size;
	int batch = 6000;
	int loops = 10;
	if (argc >= 2)
		loops = atoi(argv[1]);
	if (argc == 3)
		batch = atoi(argv[2]);
	std::cout << argc << "\t" << argv[1] << std::endl;

	// fix network architecture: input layer, hidden layer(s),
	// output layer
	const int input_size = trainingSet.imgSize * trainingSet.imgDepth;
	const int output_size = 10;
	std::vector<int> architecture({input_size, 300, output_size});

	// base output filename
	std::stringstream sname;
	sname << "mnist-" << architecture[0];
	for (size_t i = 1; i < architecture.size(); i++)
		sname << "x" << architecture[i];
	std::string basename = sname.str();

	std::ofstream ossummary(basename + "-summary");
	std::cout << "# alpha = " << alpha << "\n# lambda = " << lambda
	          << "\n# epsilon = " << epsilon << "\n# count = " << count
	          << "\n# batch = " << batch << "\n# loops = " << loops << "\n\n"
	          << "# NN Architecture: " << architecture[0];
	ossummary << "# alpha = " << alpha << "\n# lambda = " << lambda
	          << "\n# epsilon = " << epsilon << "\n# count = " << count
	          << "\n# batch = " << batch << "\n# loops = " << loops << "\n\n"
	          << "# NN Architecture: " << architecture[0];
	for (size_t i = 1; i < architecture.size(); i++)
		ossummary << " --> " << architecture[i];
	ossummary << "\n\n";

	// prepare and initialize weight matrices W_i for given
	// architecture, so that
	//
	// Z_(i+1) = W_i) * A_i
	std::vector<Mat> weights = loadWeightsMat(basename + "-weights");
	for (size_t i = 1; i < architecture.size(); i++) {
		cv::Size size = cv::Size(architecture[i - 1] + 1, architecture[i]);

		if ((weights.size() >= i) and weights[i - 1].size() == size) {
			std::cerr << "Loaded Weight Matrix[" << i << "] "
			          << weights[i - 1].t().size() << "\tWeight sum " << i << " = "
			          << cv::sum(weights[i - 1]) << std::endl;
		} else {
			Mat weight(size, CV_64F);
			cv::randu(weight, cv::Scalar::all(-epsilon), cv::Scalar::all(epsilon));
			weights.push_back(weight);
			std::cerr << "Initial Weight Matrix[" << i << "] " << weight.t().size()
			          << std::endl;
		}
	}

	// create teaching signal vectors
	std::vector<Mat> yk = makeTeachingVectors(output_size);

	std::vector<double> Jseries;

	// do mini-batch gradient descent, 10 times round
	for (int nb = 0; nb < loops * count / batch; nb++) {
		double J = 0;

		Mat Grad1 = Mat::zeros(weights[1].size(), weights[1].type());
		Mat Grad0 = Mat::zeros(weights[0].size(), weights[0].type());

#pragma omp parallel for
		for (int i = nb * batch; i < (nb + 1) * batch; i++) {
			int id = i % count;

			debuglog << "\nFeed Forward\t";
			Mat a0 = computeInputVector(trainingSet.images[id]);

			debuglog << "W0" << weights[0].t().size() << " x A0" << a0.t().size();
			Mat z1 = weights[0] * a0;
			Mat a1 = computeActivation(z1);
			debuglog << " --> W1" << weights[1].t().size() << " x A1" << a1.t().size();
			Mat z2 = weights[1] * a1;
			Mat a2 = computeOutputVector(z2);

			debuglog << "\nBack prop\t";
			// delta2 = (Theta2' * delta3)(2:hidden_layer_size+1) .*
			// sigmoidGradient(Z2);
			int label = trainingSet.labels[id];
			debuglog << "D2 = A2" << a2.t().size() << " - "
			         << " yk" << yk[label].t().size();
			Mat d2 = a2 - yk[label];
			debuglog << "(W1'" << weights[1].size() << " x " << d2.t().size()
			         << ")(2:)";
			Mat d1 = (weights[1].t() * d2)(cv::Rect(0, 1, 1, architecture[1]));
			Mat z1grad = apply(z1, z1, &sigmoidGrad);
			debuglog << d1.t().size() << " .* grad sigmoid(Z2)" << z1grad.t().size()
			         << "\n";
			d1 = d1.mul(z1grad);

#pragma omp critical
			{
				// J = J + sum(-yk.*log(A3) - (1 - yk).*(log(1 - A3))) / m;
				J += cv::sum(-1.0f * yk[label].mul(log(a2)) -
				             (1.0f - yk[label]).mul(log(1.0f - a2)))[0];

				// Delta2 = Delta2 + delta3 * A2';
				Grad1 += d2 * a1.t();

				// Delta1 = Delta1 + delta2 * A1';
				Grad0 += d1 * a0.t();
			}
		}

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
		Mat R0 = weights[0] * (lambda / batch);
		// std::cerr << " R0" << R0.t().size() << "(:,1)";
		R0(cv::Rect(0, 0, 1, architecture[1])) = Mat::zeros(1, architecture[1], CV_64F);
		Grad0 = Grad0 / batch + R0;

		Mat R1 = weights[1] * (lambda / batch);
		// std::cerr << ", R1" << R1.t().size() << "(:,1)" << "\n";
		R1(cv::Rect(0, 0, 1, architecture[2])) = Mat::zeros(1, architecture[2], CV_64F);
		Grad1 = Grad1 / batch + R1;

		double Jreg = (sqrsum(weights[0]) + sqrsum(weights[1])) * lambda / (2 * batch);
		J /= batch;

		ossummary << nb << "\tJ = " << J << " + " << Jreg << " = " << J + Jreg << std::endl;
		std::cout << nb << "\tJ = " << J << " + " << Jreg << " = " << J + Jreg << std::endl;

		weights[1] = weights[1] - alpha * Grad1;
		weights[0] = weights[0] - alpha * Grad0;

		J += Jreg;
		Jseries.push_back(J);

		size_t wndSize = count / batch;
		double JWndMean = J;
		if (Jseries.size() > wndSize) {
			JWndMean =
			    std::accumulate(Jseries.rbegin(), Jseries.rbegin() + wndSize, 0.0) /
			    wndSize;
		}
		if (J > JWndMean) {
			alpha = alpha * 0.9;
			ossummary << nb << "\tJ = " << J << "\talpha --> " << alpha << "\n";
		}
		if (alpha < 1e-14) {
			break;
		}
	}

	// store weights
	std::ofstream osweights(basename + "-weights");
	for (size_t i = 0; i < architecture.size() - 1; i++) {
		osweights << weights[i] << std::endl;
		ossummary << "Storing weights " << i << " Sum = " << cv::sum(weights[i])
		          << std::endl;
	}

	// store cost function values
	std::ofstream oscosts(basename + "-costs");
	for (size_t i = 0; i < Jseries.size(); i++)
		oscosts << i << "\t" << Jseries[i] << "\n";

	// start testing
	DatasetMNIST testset("MNIST/t10k-images-idx3-ubyte", "MNIST/t10k-labels-idx1-ubyte");
	int correct = 0;
	Mat stats = Mat::eye(10, 10, CV_64F);
	std::vector<std::pair<int, int>> errors; // image ids of wrong guesses

	for (int id = 0; id < testset.size; id++) {
		// feed forward and check
		Mat a0 = computeInputVector(testset.images[id]);
		Mat z1 = weights[0] * a0;
		Mat a1 = computeActivation(z1);
		Mat z2 = weights[1] * a1;
		Mat a2 = computeOutputVector(z2);
		int label = testset.labels[id];

		int idmax[2];
		double max;
		minMaxIdx(a2, 0, &max, 0, idmax);
		int classification = idmax[0];

		stats += yk[label] * a2.t();

		if (classification == label)
			++correct;
		else {
			errors.push_back(std::make_pair(id, classification));
		}
	}
	ossummary << "# Total correct: " << correct << " / " << testset.size << " = "
	          << static_cast<double>(correct) / testset.size << std::endl;

	std::cout << "# Total correct: " << correct << " / " << testset.size << " = "
	          << static_cast<double>(correct) / testset.size
	          << "\n# Total classification error: "
	          << static_cast<double>(testset.size - correct) / testset.size << std::endl;

	ossummary << "# row: label, column: estimate\n" << stats / testset.size << std::endl;

	std::string title = "classification errors";
	cv::namedWindow(title);
	Mat display = montageList(testset, errors);
	cv::imshow(title, display);
	while (cv::waitKey(0) != 27)
		;
}

double sqrsum(const Mat &src)
{
	assert(src.isContinuous());
	const double *x = src.ptr<double>(0);
	cv::Size size = src.size();
	double sum = 0;
#pragma omp parallel for reduction(+ : sum)
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
		std::cerr << "k = " << k << " --> yk'" << yk[k].size() << " = " << yk[k].t()
		          << "\t y" << k << "(" << k << ") = " << yk[k].at<double>(0, k) << "\n";
	cv::waitKey(-1);
#endif

	return yk;
}
