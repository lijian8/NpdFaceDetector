#pragma once

#include <array>
#include <vector>
#include <cassert>
#include <utility>
#include <iostream>
#include <numeric>
#include <deque>

#include <ppl.h>
#define NOMINMAX
#include <amp.h>
#include <amp_math.h>

namespace fm = concurrency::fast_math;
namespace pm = concurrency::precise_math;

enum LabelsEnum {
	NEGATIVE = 0,
	POSITIVE = 1,
	N_LABELS = 2
};

class Sample {
public:
	Sample(int label, std::vector<unsigned char> const && pixels) :
		_label(label), _weight(1.0), _pixels(pixels) {
		assert(label == NEGATIVE || label == POSITIVE);
	}

	Sample(int label, std::vector<unsigned char> const && pixels, double weight) :
		Sample(label, std::move(pixels)) {
		set_weight(weight);
	}

	int get_label() const { return _label; }
	auto const& get_pixels() const { return _pixels; }

	void set_weight(double weight) { _weight = weight; }
	double get_weight() const { return _weight; }

private:
	int _label;
	double _weight;
	std::vector<unsigned char> _pixels;

	friend void swap(Sample &, Sample &);
};

// Needed by std::partition during learning.
void swap(Sample & first, Sample & second) {
	std::swap(first._label, second._label);
	std::swap(first._weight, second._weight);
	std::swap(first._pixels, second._pixels);
}

// Convert a pair of indices to a linear indices. 'first' must be less than 'second'.
int pair_to_linear(std::pair<int, int> pair) {
	int n = pair.first;
	int m = pair.second;
	if (!(n >= 0 && n < m))
		return 0;
	assert(n >= 0 && n < m);
	return m * (m - 1) / 2 + n;
}

// Convert a linear index to a pair of indices. Slow.
std::pair<int, int> linear_to_pair(int linear) {
	assert(linear >= 0);
	int m = static_cast<int>(std::floor(0.5f * (1.0f + std::sqrt(8.0f * static_cast<float>(linear) + 1.0f))));
	int n = linear - (m * (m - 1)) / 2;
	assert(n >= 0 && n < m);
	return std::make_pair(n, m);
}

void linear_to_pair(int * n, int * m, int linear) restrict(amp) {
	*m = static_cast<int>(fm::floor(0.5f * (1.0f + fm::sqrt(8.0f * static_cast<float>(linear) + 1.0f))));
	*n = linear - (*m * (*m - 1)) / 2;
}

int get_feature_dim(int pixel_count) {
	return pair_to_linear(std::make_pair(pixel_count - 2, pixel_count - 1)) + 1;
}

// We define the normalized pixel difference as
//   (n - m) / (n + m)
// which has a range of [-1, 1].
// But we want the values to be in the [0, 256) range.
// So we add one, divide by two and multiply by 256
//  ((n - m) / (n + m) + 1) / 2 * 256 == n / (n + m) * 256
//
unsigned int get_npd(unsigned int n, unsigned int m) restrict(amp, cpu) {
	float diff = 0.5f;
	if (n > 0 || m > 0) {
		// Without the 0.5, things like 256 * 33 / (33 + 33) will floor to 
		// 128 on cpu and 127 on gpu.
		diff = static_cast<float>(n) / (static_cast<float>(n + m) + 0.5f);
	}
	// This is correct. If you do round(255 * diff), the bins at
	// 0 and 255 will only be 0.5 wide.
	diff = fm::fmin(fm::floor(256.0f * diff), 255.0f);
	return static_cast<unsigned int>(diff);
}

auto compute_npd_table() {
	std::array<std::array<unsigned char, 256>, 256> npd_table;
	for (int m = 0; m < 256; ++m) {
		for (int n = 0; n < 256; ++n) {
			npd_table[n][m] = get_npd(n, m);
		}
	}
	return npd_table;
}

// Using a table is three times faster on cpu.
static const auto g_npd_table = compute_npd_table();

unsigned char get_feature_value(std::vector<unsigned char> const & pixels, std::pair<int, int> pixel_indices) {
	return g_npd_table[pixels[pixel_indices.first]][pixels[pixel_indices.second]];
}

bool goes_left(unsigned char feature_value, unsigned char threshold) {
	// Warning: compute_best_split() depends, implicitly, on this behaviour.
	return (feature_value <= threshold);
}

class Split {
public:
	Split(std::pair<int, int> pixel_indices, int threshold) :
		_pixel_indices(pixel_indices), _threshold(threshold) {
		assert(pixel_indices.first >= 0 && pixel_indices.second >= 0);
		assert(threshold >= 0 && threshold < 256);
	}
	std::pair<int, int> get_pixel_indices() const {
		return _pixel_indices;
	}
	unsigned char get_threshold() const {
		return _threshold;
	}
	unsigned char get_feature_value(std::vector<unsigned char> const & pixels) const {
		return ::get_feature_value(pixels, _pixel_indices);
	}
	bool goes_left(std::vector<unsigned char> const & pixels) const {
		return ::goes_left(get_feature_value(pixels), _threshold);
	}

	static Split Dummy() {
		// Make sure that _everything_ goes left.
		assert(::goes_left(255, 255));
		return Split({ 0, 1 }, 255);
	}

private:
	std::pair<int, int> _pixel_indices;
	unsigned char _threshold;
};

double pow2(double x) {
	return x * x;
}

class SampleRange {
public:
	typedef Sample * iterator;
	typedef Sample const * const_iterator;

	SampleRange(Sample * begin, Sample * end) :
		_begin(begin), _end(end) {
	}
	SampleRange(std::vector<Sample> & samples) :
		_begin(samples.data()), _end(samples.data() + samples.size()) {
	}

	int size() const { return std::distance(begin(), end()); }
	bool empty() const { return (size() == 0); }

	iterator begin() { return _begin; }
	iterator end() { return _end; }

	const_iterator begin() const { return _begin; }
	const_iterator end() const { return _end; }

private:
	Sample * _begin;
	Sample * _end;
};


void sort_lowest_weight_first(SampleRange * sample_range) {
	std::sort(sample_range->begin(), sample_range->end(),
		[](Sample const & lhs, Sample const & rhs) {
		return (lhs.get_weight() < rhs.get_weight());
	});
}

// For each feature, make a histogram based on the sample weights.
auto accumulate_histograms_cpu(SampleRange * sample_range) {

	assert(!sample_range->empty());
	int const pixel_count = sample_range->begin()->get_pixels().size();
	int const feature_dim = get_feature_dim(pixel_count);
	std::vector<std::array<float, 256>> histograms(feature_dim);
	for (auto & h : histograms) {
		h.fill(0.0);
	}

	// So that add floats of approx. equal size.
	sort_lowest_weight_first(sample_range);

	for (auto sample = sample_range->begin(); sample < sample_range->end(); ++sample) {
		auto const weight = static_cast<float>(sample->get_weight());
		auto const & pixels = sample->get_pixels();
		concurrency::parallel_for(1, pixel_count, [&](int m) {
			for (int n = 0; n < m; ++n) {
				auto pixel_indices = std::make_pair(n, m);
				int const feature_index = pair_to_linear(pixel_indices);
				unsigned char const bin = get_feature_value(pixels, pixel_indices);
				histograms[feature_index][bin] += weight;
			}
		});
	}

	return histograms;
}

auto accumulate_histograms_gpu(SampleRange * sample_range) {

	assert(!sample_range->empty());
	int const sample_count = sample_range->size();
	int const pixel_count = sample_range->begin()->get_pixels().size();
	int const feature_dim = get_feature_dim(pixel_count);

	double weight_sum = 0.0;
	std::for_each(sample_range->begin(), sample_range->end(), [&](Sample const & sample) { weight_sum += sample.get_weight(); });
	double scale = std::numeric_limits<unsigned int>::max() / weight_sum;
	std::cout << "scale = " << scale << "\n";

	// Add dummy samples to get a multiple of 4.
	int const extended_sample_count = 4 * ((sample_count + 3) / 4);
	std::vector<unsigned char> pixel_stage(extended_sample_count * pixel_count);
	std::vector<unsigned int> weight_stage(extended_sample_count);
	auto samples = sample_range->begin();
	for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
		weight_stage[sample_index] = static_cast<unsigned int>(scale * samples[sample_index].get_weight());
		for (int pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
			pixel_stage[pixel_index * extended_sample_count + sample_index] = samples[sample_index].get_pixels()[pixel_index];
		}
	}
	// Set dummy sample weights to 0.
	for (int sample_index = sample_count; sample_index < extended_sample_count; ++sample_index) {
		weight_stage[sample_index] = 0;
	}

	std::vector<unsigned int> histograms(256 * feature_dim);

	int const pixel_block_count = extended_sample_count / 4;
	concurrency::array_view<const unsigned int, 2> pixel_view(pixel_count, pixel_block_count, reinterpret_cast<unsigned int *>(pixel_stage.data()));
	concurrency::array_view<const unsigned int, 1> weight_view(weight_stage);
	concurrency::array_view<unsigned int, 2> histograms_view(feature_dim, 256, histograms.data());
	histograms_view.discard_data();

	static int const MAX_FEATURES_PER_KERNEL = 65535;
	static int const TILE_SIZE = 256;

	int const feature_block_count = (feature_dim + MAX_FEATURES_PER_KERNEL - 1) / MAX_FEATURES_PER_KERNEL;
	int const feature_block_size = (feature_dim + feature_block_count - 1) / feature_block_count;

	for (int feature_block_index = 0; feature_block_index < feature_block_count; ++feature_block_index) {

		int const feature_start_index = feature_block_index * feature_block_size;
		int const remaining_features_count = feature_dim - feature_start_index;
		int const actual_feature_block_size = std::min(feature_block_size, remaining_features_count);

		concurrency::extent<2> extent(TILE_SIZE, actual_feature_block_size);
		concurrency::parallel_for_each(extent.tile<TILE_SIZE, 1>(),
			[=](concurrency::tiled_index<TILE_SIZE, 1> index) restrict(amp) {
			tile_static unsigned int tile_histogram[256];

			tile_histogram[index.global[0]] = 0;
			index.barrier.wait_with_tile_static_memory_fence();

			int feature_index = feature_start_index + index.global[1];
			if (feature_index < feature_dim) {
				int n, m;
				linear_to_pair(&n, &m, feature_index);

				int begin_index = 0;
				while (begin_index < pixel_block_count) {
					int pixel_block_index = begin_index + index.global[0];
					unsigned int pixels_n = pixel_view[n][pixel_block_index];
					unsigned int pixels_m = pixel_view[m][pixel_block_index];

					if (pixel_block_index < pixel_block_count) {
						for (int pixel_index = 0; pixel_index < 4; ++pixel_index) {
							unsigned int pixel_n = 0xff & pixels_n;
							unsigned int pixel_m = 0xff & pixels_m;

							int bin = get_npd(pixel_n, pixel_m);

							int sample_index = 4 * pixel_block_index + pixel_index; // Little endian
							concurrency::atomic_fetch_add(&tile_histogram[bin], weight_view[sample_index]);

							pixels_n >>= 8;
							pixels_m >>= 8;
						}
					}
					begin_index += TILE_SIZE;
				}
			}

			index.barrier.wait_with_tile_static_memory_fence();
			if (feature_index < feature_dim) {
				histograms_view[feature_index][index.global[0]] = tile_histogram[index.global[0]];
			}
		});
	}
	histograms_view.synchronize();

	std::vector<std::array<float, 256>> histograms_out(feature_dim);
	for (int i = 0; i < feature_dim; ++i) {
		int sum = 0;
		for (int j = 0; j < 256; ++j) {
			histograms_out[i][j] = static_cast<float>(histograms[i * 256 + j]) / static_cast<float>(scale);
			sum += histograms[i * 256 + j];
		}
	}

	return histograms_out;
}

// Indexing is [label][feature_index][bin]
typedef std::array<std::vector<std::array<float, 256>>, N_LABELS> histograms_type;

auto sort_positives_first_and_get_midpoint(SampleRange * sample_range) {
	return std::partition(sample_range->begin(), sample_range->end(),
		[](Sample const & sample) {
		return (sample.get_label() == POSITIVE);
	});
}

// Compute the total weight per label.
std::array<double, N_LABELS> compute_weight_totals(histograms_type const & histograms) {
	std::array<double, N_LABELS> weight_totals;
	for (int label = 0; label < N_LABELS; ++label) {
		auto const & first_histogram = histograms[label].front();
		weight_totals[label] = std::accumulate(first_histogram.begin(), first_histogram.end(), 0.0);
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
Split compute_best_split(histograms_type const & histograms) {
	assert(!histograms[NEGATIVE].empty() &&
		histograms[NEGATIVE].size() == histograms[POSITIVE].size());

	enum {
		LEFT = 0,
		RIGHT = 1,
		N_SIDES = 2
	};

	auto const feature_dim = histograms[NEGATIVE].size();
	auto const weight_totals = compute_weight_totals(histograms);

	int best_feature_index = -1;
	int best_threshold = 0;
	double best_score = 0.0;
	for (size_t feature_index = 0; feature_index < feature_dim; ++feature_index) {
		std::array<std::array<double, N_LABELS>, N_SIDES> weight_sums;
		weight_sums[LEFT].fill(0.0);
		for (int threshold = 0; threshold < 256; ++threshold) {

			// Assumes that anything strictly above the threshold goes right.
			for (int label = 0; label < N_LABELS; ++label) {
				weight_sums[LEFT][label] += histograms[label][feature_index][threshold];
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
		return Split::Dummy();
	}
	return Split(linear_to_pair(best_feature_index), best_threshold);
}

Split fit_stump(SampleRange * sample_range) {
	auto const range_midpoint = sort_positives_first_and_get_midpoint(sample_range);
	auto positive_range = SampleRange(sample_range->begin(), range_midpoint);
	auto negative_range = SampleRange(range_midpoint, sample_range->end());
	if (positive_range.empty() || negative_range.empty()) {
		std::cout << "Warning: Node is pure, can not split.\n";
		return Split::Dummy();
	}
	histograms_type histograms;
	histograms[POSITIVE] = accumulate_histograms_cpu(&positive_range);
	histograms[NEGATIVE] = accumulate_histograms_cpu(&negative_range);
	return compute_best_split(histograms);
}

SampleRange::iterator sort_goes_left_first_and_get_midpoint(SampleRange * range, Split const& split) {
	return std::partition(range->begin(), range->end(),
		[split](Sample & sample) { return split.goes_left(sample.get_pixels()); });
}

int get_split_count(int depth) {
	assert(depth > 0);
	return (1 << depth) - 1;
}

int get_leaf_count(int depth) {
	return get_split_count(depth) + 1;
}

class DenseTree {
public:
	DenseTree(int depth) : _depth(depth) {
		assert(depth > 0);
		_splits.reserve(get_split_count(depth));
		_leafs.reserve(get_leaf_count(depth));
	};
	void push_back_split(Split const & split) {
		return _splits.push_back(split);
	}
	void push_back_leaf(double leaf) {
		return _leafs.push_back(leaf);
	}
	bool is_complete() const {
		return (_splits.size() == get_split_count(_depth) &&
			_leafs.size() == get_leaf_count(_depth));
	}
	int get_depth() const {
		assert(is_complete());
		return _depth;
	}
	double predict(std::vector<unsigned char> const& pixels) const {
		assert(is_complete());
		int index = 0;
		int split_count = get_split_count(_depth);
		while (index < split_count) {
			index = get_child_index(index, _splits[index].goes_left(pixels));
		}
		index -= split_count;
		return _leafs[index];
	}

private:
	int _depth;
	std::vector<Split> _splits;
	std::vector<double> _leafs;

	int get_child_index(int index, bool goes_left) const {
		return index * 2 + (goes_left ? 1 : 2);
	}
};

double get_weighted_mean(SampleRange const & range) {
	double w_sum = 0.0;
	double wy_sum = 0.0;
	for (auto it = range.begin(); it != range.end(); ++it) {
		w_sum += it->get_weight();
		wy_sum += (2 * it->get_label() - 1) * it->get_weight();
	}
	if (w_sum > 0.0) {
		return wy_sum / w_sum;
	}
	return 0.0;
}

DenseTree fit_tree(SampleRange * range, int depth, int min_leaf_occupancy) {
	assert(depth > 0);
	assert(min_leaf_occupancy >= 0);

	std::deque<SampleRange> range_stack;
	range_stack.push_back(*range);
	size_t const leaf_count = get_leaf_count(depth);

	// Walk the tree in breadth-first order.
	DenseTree tree(depth);
	while (range_stack.size() < leaf_count) {
		auto range = range_stack.front();
		range_stack.pop_front();
		Split split = fit_stump(&range);
		tree.push_back_split(split);
		auto range_midpoint = sort_goes_left_first_and_get_midpoint(&range, split);
		range_stack.push_back(SampleRange(range.begin(), range_midpoint));
		range_stack.push_back(SampleRange(range_midpoint, range.end()));
	}

	// The samples are now partitioned into one range per leaf.
	while (!range_stack.empty()) {
		auto range = range_stack.front();
		range_stack.pop_front();
		if (range.size() >= min_leaf_occupancy) {
			auto weighted_mean = get_weighted_mean(range);
			tree.push_back_leaf(weighted_mean);
		}
		else {
			tree.push_back_leaf(0.0);
		}
	}
	assert(tree.is_complete());

	return tree;
}
