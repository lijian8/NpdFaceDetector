#pragma once

#include <array>
#include <vector>
#include <cassert>
#include <utility>
#include <iostream>
#include <numeric>

enum LabelsEnum {
	NEGATIVE = 0,
	POSITIVE = 1,
	N_LABELS = 2
};

class Sample {
public:
	Sample(int label, std::vector<unsigned char> const && features) :
		_label(label), _weight(0.0), _features(features) {
		assert(label == NEGATIVE || label == POSITIVE);
	}

	Sample(int label, std::vector<unsigned char> const && features, double weight) :
		Sample(label, std::move(features)) {
		set_weight(weight);
	}

	int get_label() const { return _label; }
	auto const& get_features() const { return _features; }
	
	void set_weight(double weight) { _weight = weight; }
	double get_weight() const { return _weight; }

private:
	int _label;
	double _weight;
	std::vector<unsigned char> _features;

	friend void swap(Sample &, Sample &);
};

void swap(Sample & first, Sample & second) {
	std::swap(first._label, second._label);
	std::swap(first._weight, second._weight);
	std::swap(first._features, second._features);
}

// Convert a pair of indices to a linear indices. 'first' must be less than 'second'.
int pair_to_linear(std::pair<int, int> pair) {
	assert(pair.first >= 0);
	assert(pair.first < pair.second);
	int n = pair.first;
	int m = pair.second;
	return m * (m - 1) / 2 + n;
}

// Convert a linear index to a pair of indices. Slow.
std::pair<int, int> linear_to_pair(int linear) {
	assert(linear >= 0);
	int m = static_cast<int>(std::floor(0.5 * (1.0 + std::sqrt(8.0 * static_cast<double>(linear) + 1))));
	int n = linear - m * (m - 1) / 2;
	return std::make_pair(n, m);
}

class Split {
public:
	Split(int feature_index, int threshold) : _feature_index(feature_index), _threshold(threshold) {
		assert(feature_index >= 0);
		assert(threshold >= 0 && threshold < 256);
	}
	int get_feature_index() const { return _feature_index; }
	unsigned char get_threshold() const { return _threshold; }
	bool goes_left(std::vector<unsigned char> const & features) const { return (features[_feature_index] <= _threshold); }

private:
	int _feature_index;
	unsigned char _threshold;
};

double pow2(double x) {
	return x * x;
}

class SampleRange {
public:
	typedef Sample * iterator;
	typedef Sample const * const_iterator;

	SampleRange(Sample * begin, Sample * end) : _begin(begin), _end(end) {}
	SampleRange(std::vector<Sample> & samples) : _begin(samples.data()), _end(samples.data() + samples.size()) {}

	iterator begin() { return _begin;  }
	iterator end() { return _end; }

	const_iterator begin() const { return _begin; }
	const_iterator end() const { return _end; }

private:
	Sample * _begin;
	Sample * _end;
};

typedef std::array<std::array<double, 256>, N_LABELS> histogram_type;

// For each feature, make a histogram based on the sample weights,
// one for positive and one for negative samples.
// Indexing is [feature_index][label][feature_value]
//
std::vector<histogram_type> accumulate_histograms(SampleRange const & range) {

	size_t const feature_dimension = range.begin()->get_features().size();
	std::vector<histogram_type> histograms(feature_dimension);
	for (auto & h : histograms) {
		h[NEGATIVE].fill(0.0);
		h[POSITIVE].fill(0.0);
	}

	for (auto sample = range.begin(); sample < range.end(); ++sample) {
		auto label = sample->get_label();
		auto weight = sample->get_weight();
		auto const& features = sample->get_features();
		for (size_t feature_index = 0; feature_index < feature_dimension; ++feature_index) {
			histograms[feature_index][label][features[feature_index]] += weight;
		}
	}

	return histograms;
}

std::array<double, N_LABELS> compute_weight_totals(histogram_type const & histogram) {
	std::array<double, N_LABELS> weight_totals;
	for (int label = 0; label < N_LABELS; ++label) {
		weight_totals[label] = std::accumulate(histogram[label].begin(), histogram[label].end(), 0.0);
	}
	return weight_totals;
}

// Analyze the histograms to pick the best split.
// The goal is to minimize
//
// L = sum(w_k * (y_k - m_L) ^ 2, k in L) + 
//     sum(w_k * (y_k - m_R) ^ 2, k in R)
//
// where L/R are the left and right subsets and 
// m_L/m_R are the weighted means for those subsets.
// It can be shown that this is the same as maximizing
// 
// S = sum(w_k * y_k, k in L) ^ 2 / sum(w_k, k in L) +
//     sum(w_k * y_k, k in R) ^ 2 / sum(w_k, k in R)
//
Split compute_best_split(std::vector<histogram_type> const & histograms) {
	assert(!histograms.empty());

	enum {
		LEFT = 0,
		RIGHT = 1,
		N_SIDES = 2
	};

	auto const weight_totals = compute_weight_totals(histograms.front());
	int best_feature_index = -1;
	int best_threshold = 0;
	double best_score = 0.0;
	for (size_t feature_index = 0; feature_index < histograms.size(); ++feature_index) {
		histogram_type const & histogram = histograms[feature_index];
		std::array<std::array<double, N_LABELS>, N_SIDES> weight_sums;
		weight_sums[LEFT].fill(0.0);
		for (int threshold = 0; threshold < 256; ++threshold) {

			// Assumes that anything strictly above the threshold goes right.
			for (int label = 0; label < N_LABELS; ++label) {
				weight_sums[LEFT][label] += histogram[label][threshold];
				weight_sums[RIGHT][label] = weight_totals[label] - weight_sums[LEFT][label];
			}

			std::array<double, N_SIDES> wy_sums;
			std::array<double, N_SIDES> w_sums;
			for (int side = 0; side < N_SIDES; ++side) {
				wy_sums[side] = weight_sums[side][POSITIVE] - weight_sums[side][NEGATIVE];
				w_sums[side] = weight_sums[side][POSITIVE] + weight_sums[side][NEGATIVE];
			}

			bool const all_weights_positive =
				std::all_of(w_sums.begin(), w_sums.end(), [](double const& x) { return (x > 0.0); });

			if (all_weights_positive) {
				double const score = pow2(wy_sums[LEFT]) / w_sums[LEFT] + pow2(wy_sums[RIGHT]) / w_sums[RIGHT];
				if (score > best_score) {
					best_feature_index = feature_index;
					best_threshold = threshold;
					best_score = score;
				}
			}
		} // for threshold
	} // for feature_index

	if (best_feature_index < 0) {
		std::cout << "Warning: No feature splits the samples.\n";
		return Split(0, 0);
	}
	return Split(best_feature_index, best_threshold);
}

Split fit_stump(SampleRange const & range) {
	auto const histograms = accumulate_histograms(range);
	return compute_best_split(histograms);
}

SampleRange::iterator partition_samples(SampleRange * range, Split const& split) {
	return std::partition(range->begin(), range->end(),
		[split](Sample & sample) { return split.goes_left(sample.get_features()); });
}

class DenseTree {
public:
	DenseTree() {};
private:
	int depth;
	std::vector<Split> _splits;
	std::vector<double> _leafs;
};
