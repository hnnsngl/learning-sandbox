#include "loadMNIST.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <cmath>

cv::Mat rotateMinArea(cv::Mat &img)
{
	img.convertTo(img, CV_32F);

	cv::Mat src(img.size(), img.type());
	img.convertTo(src, CV_32F);

	cv::threshold(src, src, 0.8, 1.0, cv::THRESH_BINARY);

	std::vector<cv::Point> points;
	for (auto it = src.begin<double>(); it != src.end<double>(); it++)
		if (*it)
			points.push_back(it.pos());

	cv::RotatedRect rect = cv::minAreaRect(points);
	double scale = std::min(28.0 / rect.size.height, 28.0 / rect.size.width);
	double angle = rect.angle;
	if (angle < -45.0)
		angle += 90;
	std::cerr << "rotation: " << angle << "\n";

	cv::Mat rot = cv::getRotationMatrix2D(rect.center, angle, 1.0);
	cv::Mat dst = cv::Mat::zeros(img.size(), img.type());
	cv::warpAffine(img, dst, rot, cv::Size(28, 28));
	return dst;
}

// def deskew(image):
//     c,v = moments(image)
//     alpha = v[0,1]/v[0,0]
//     affine = array([[1,0],[alpha,1]])
//     ocenter = array(image.shape)/2.0
//     offset = c-dot(affine,ocenter)
//     return interpolation.affine_transform(image,affine,offset=offset)

cv::Mat deskewedMoments(const cv::Mat &img)
{
	cv::Moments moments = cv::moments(img.reshape(1, 28));
	double alpha = asin(moments.m00 / moments.m01) / 3.14 * 180.0;
	std::cout << "deskew angle: " << alpha << "\n";

	cv::Mat rot = cv::getRotationMatrix2D(cv::Point2d(28 / 2, 28 / 2), alpha, 1.0);
	cv::Mat dst = cv::Mat::zeros(img.size(), img.type());
	cv::warpAffine(img, dst, rot, cv::Size(28, 28));
	return dst;
}

int main()
{
	DatasetMNIST dataset("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte");
	cv::namedWindow("deskewing");
	for (int i = 0; i < dataset.size; i++) {

		cv::Mat img = dataset.images[i];

		std::vector<cv::Mat> dst;
		dst.push_back(img);
		dst.push_back(rotateMinArea(img));
		dst.push_back(deskewedMoments(img));

		cv::Size size = img.size();
		size.width *= dst.size();
		cv::Mat montage = cv::Mat::zeros(size, img.type());

		for (int i = 0; i < dst.size(); i++)
			dst[i].copyTo(montage(cv::Rect(i * 28, 0, 28, 28)));

		cv::imshow("deskewing", montage);
		int key = cv::waitKey(0);
		if (key == 27)
			break; // stop on [ESC]
	}
}
