#pragma once

#include <vector>
#include <string>

// most general would be some DatasetIterator, which could be used on
// in-memory datasets as well as disk-based ones

template <typename T> class DatasetItem
{
	// ...
}

template <typename T>
class Dataset
{
      public:
	/** representation type, such as cv::Mat, or some type
	 *  from BLAS/LAPACK */
	typedef T Mat;

	Dataset(const Dataset);

	/** return size of dataset */
	int size() const;

	/** fetch data item in input vector represenation */
	Mat get(int i) const;

	/** fetch data item for visual representation */
	Mat getVisual(int i) const;

	/** load items from file into dataset
	 *  @param	filename
	 *  @return 	number of loaded items */
	int load(std::string filename) virtual = 0;

	/** return the number of items with the given label */
	int countLabel(int label) const;

      private:
	std::vector<Mat> items;
	std::vector<int> labels;
	std::map<int, int> counts;
};

template <typename T> int Dataset<T>::size() const { return items.size(); }

template <typename T> T Dataset<T>::get(int i) const { return items[i]; }

template <typename T> T Dataset<T>::getVisual(int i) const { return items[i]; }

template <typename T> int Dataset::countLabel(int label) const
{
	if (counts.find(label) != end(counts))
		return counts[label];

	int count = 0;
	for (const auto &it : labels)
		if (it == label)
			++count;
	return count;
}
