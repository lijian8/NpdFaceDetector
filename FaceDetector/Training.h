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

enum LabelsEnum {
	NEGATIVE = 0,
	POSITIVE = 1,
	N_LABELS = 2
};

class Sample {
public:
	Sample(int label, std::vector<unsigned char> const && pixels) :
		_label(label), _prediction(0.0), _weight(1.0), _pixels(pixels) {
		assert(label == NEGATIVE || label == POSITIVE);
	}

	Sample(int label, std::vector<unsigned char> const && pixels, double weight) :
		Sample(label, std::move(pixels)) {
		set_weight(weight);
	}

	int get_label() const { return _label; }
	int get_signed_label() const { return 2 * _label - 1; }

	double get_prediction() const { return _prediction; }
	void set_prediction(double prediction) { _prediction = prediction; }

	double get_weight() const { return _weight; }
	void set_weight(double weight) { _weight = weight; }

	auto const& get_pixels() const { return _pixels; }

private:
	int _label;
	double _prediction;
	double _weight;
	std::vector<unsigned char> _pixels;
};

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
	typedef Sample & reference;
	typedef Sample const & const_reference;

	SampleRange() :
		_begin(nullptr), _end(nullptr) {
	}
	SampleRange(Sample * begin, Sample * end) :
		_begin(begin), _end(end) {
	}
	SampleRange(std::vector<Sample> & samples) :
		_begin(samples.data()), _end(samples.data() + samples.size()) {
	}

	int size() const { return std::distance(begin(), end()); }
	bool empty() const { return (size() == 0); }

	reference operator[] (int idx) {
		assert(0 <= idx && idx < size());
		return _begin[idx];
	}
	const_reference operator[] (int idx) const {
		assert(0 <= idx && idx < size());
		return _begin[idx];
	}

	iterator begin() { return _begin; }
	const_iterator begin() const { return _begin; }

	iterator end() { return _end; }
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

void sort_highest_weight_first(SampleRange * sample_range) {
	std::sort(sample_range->begin(), sample_range->end(),
		[](Sample const & lhs, Sample const & rhs) {
		return (lhs.get_weight() > rhs.get_weight());
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

	// The gpu can only accumulate uints, so we need to convert the weights.
	// We pick a scaling so that we approximatly hit uint max if all samples fall in the same bin.
	double weight_sum = 0.0;
	for (auto const & sample : *sample_range) { weight_sum += sample.get_weight(); }
	double const scale = 0.99 * (std::numeric_limits<unsigned int>::max() / weight_sum);

	// The gpu does not support chars, so we pack four pixels into a
	// uint. We add up to three dummy samples, to get an even multiple of four.
	int const extended_sample_count = 4 * ((sample_count + 3) / 4);
	std::vector<unsigned char> pixel_stage(extended_sample_count * pixel_count);
	std::vector<unsigned int> weight_stage(extended_sample_count);
	auto samples = sample_range->begin();
	for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
		weight_stage[sample_index] = static_cast<unsigned int>(std::round(scale * samples[sample_index].get_weight()));
		for (int pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
			pixel_stage[pixel_index * extended_sample_count + sample_index] = samples[sample_index].get_pixels()[pixel_index];
		}
	}
	// Set dummy samples' weights to 0.
	for (int sample_index = sample_count; sample_index < extended_sample_count; ++sample_index) {
		weight_stage[sample_index] = 0;
	}

	// Use a raw array instead of a vector, to avoid zero initialization.
	auto histograms = std::unique_ptr<unsigned int[]>(new unsigned int[256 * feature_dim]);

	// Setup the array views.
	int const pixel_block_count = extended_sample_count / 4;
	concurrency::array_view<const unsigned int, 2> pixel_view(pixel_count, pixel_block_count, reinterpret_cast<unsigned int *>(pixel_stage.data()));
	concurrency::array_view<const unsigned int, 1> weight_view(weight_stage);
	concurrency::array_view<unsigned int, 2> histograms_view(feature_dim, 256, histograms.get());
	histograms_view.discard_data();

	// We can't spawn enough threads to handle all features in one kernel,
	// so we split the features up into "feature_blocks".
	int const MAX_FEATURES_PER_KERNEL = 65535;
	int const feature_block_count = (feature_dim + MAX_FEATURES_PER_KERNEL - 1) / MAX_FEATURES_PER_KERNEL;
	int const feature_block_size = (feature_dim + feature_block_count - 1) / feature_block_count;

	for (int feature_block_index = 0; feature_block_index < feature_block_count; ++feature_block_index) {

		int const feature_start_index = feature_block_size * feature_block_index;
		int const remaining_features_count = feature_dim - feature_start_index;
		int const actual_feature_block_size = std::min(feature_block_size, remaining_features_count);

		static int const TILE_SIZE = 256;
		concurrency::extent<2> extent(TILE_SIZE, actual_feature_block_size);
		concurrency::parallel_for_each(extent.tile<TILE_SIZE, 1>(),
			[=](concurrency::tiled_index<TILE_SIZE, 1> index) restrict(amp) {

			// Each tile fills one histogram.
			tile_static unsigned int tile_histogram[256];
			tile_histogram[index.global[0]] = 0;
			index.barrier.wait_with_tile_static_memory_fence();

			int const feature_index = feature_start_index + index.global[1];
			int n, m;
			linear_to_pair(&n, &m, feature_index);

			// Process 'TILE_SIZE' pixel blocks in parallel.
			for (int pixel_block_start = 0; pixel_block_start < pixel_block_count; pixel_block_start += TILE_SIZE) {
				int const pixel_block_index = pixel_block_start + index.global[0];

				// The 'pixel_block_count' is not necessarily a multiple of 'TILE_SIZE'.
				if (pixel_block_index < pixel_block_count) {
					unsigned int const pixel_block_n = pixel_view[n][pixel_block_index];
					unsigned int const pixel_block_m = pixel_view[m][pixel_block_index];

					// There are four pixels in each pixel block.
					for (int pixel_index = 0; pixel_index < 4; ++pixel_index) {
						unsigned int const pixel_n = (pixel_block_n >> 8 * pixel_index) & 0xff;
						unsigned int const pixel_m = (pixel_block_m >> 8 * pixel_index) & 0xff;

						int const bin = get_npd(pixel_n, pixel_m);

						// We are little endian, so the LSB is first in memory.
						int const sample_index = 4 * pixel_block_index + pixel_index;
						concurrency::atomic_fetch_add(&tile_histogram[bin], weight_view[sample_index]);
					}
				}
			}

			// Update global memory with this tiles histogram.
			index.barrier.wait_with_tile_static_memory_fence();
			histograms_view[feature_index][index.global[0]] = tile_histogram[index.global[0]];
		});
	} // for feature_block_index

	// Convert the histograms to floats. This will unfortunately zero-fill
	// the histograms. That seems hard to avoid, but we make sure to do the
	// work before waiting for syncronization.
	std::vector<std::array<float, 256>> histograms_out(feature_dim);
	histograms_view.synchronize();

	for (int i = 0; i < feature_dim; ++i) {
		for (int j = 0; j < 256; ++j) {
			histograms_out[i][j] = static_cast<float>(histograms[i * 256 + j]) / static_cast<float>(scale);
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

			bool const all_weights_positive = (w_sums[LEFT] > 0.0 && w_sums[RIGHT] > 0.0);

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

auto get_partitioned_ranges(SampleRange * sample_range) {
	auto range_midpoint = sort_positives_first_and_get_midpoint(sample_range);
	std::array<SampleRange, N_LABELS> partition;
	partition[POSITIVE] = SampleRange(sample_range->begin(), range_midpoint);
	partition[NEGATIVE] = SampleRange(range_midpoint, sample_range->end());
	return partition;
}

Split fit_stump(SampleRange * sample_range) {
	auto partitioned_ranges = get_partitioned_ranges(sample_range);
	if (partitioned_ranges[POSITIVE].empty() ||
		partitioned_ranges[NEGATIVE].empty()) {
		std::cout << "Warning: Node is pure, can not split.\n";
		return Split::Dummy();
	}
	histograms_type histograms;
	histograms[POSITIVE] = accumulate_histograms_gpu(&partitioned_ranges[POSITIVE]);
	histograms[NEGATIVE] = accumulate_histograms_gpu(&partitioned_ranges[NEGATIVE]);
	return compute_best_split(histograms);
}

SampleRange::iterator sort_goes_left_first_and_get_midpoint(SampleRange * range, Split const & split) {
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
		wy_sum += it->get_weight() * it->get_signed_label();
	}
	if (w_sum > 0.0) {
		return wy_sum / w_sum;
	}
	return 0.0;
}

DenseTree fit_tree(SampleRange * range, int depth, int min_leaf_occupancy) {
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
			std::cout << "Warning: Terminal node is starved of samples.\n";
			tree.push_back_leaf(0.0);
		}
	}
	assert(tree.is_complete());

	return tree;
}

template <typename T>
T const & clamp(T const & value, T const & min, T const & max) {
	return std::min(std::max(value, min), max);
}

void normalize_weights_for_single_label(SampleRange * sample_range) {
	if (sample_range->empty()) {
		return;
	}

	double weight_sum = 0.0;
	int active_label = sample_range->begin()->get_label();
	for (auto const & sample : *sample_range) {
		assert(sample.get_label() == active_label);
		weight_sum += sample.get_weight();
	}

	if (weight_sum > 0.0) {
		double const scale = 1.0 / weight_sum;
		for (auto & sample : *sample_range) {
			sample.set_weight(scale * sample.get_weight());
		}
	}
	else {
		for (auto & sample : *sample_range) {
			double const weight = 1.0 / sample_range->size();
			sample.set_weight(weight);
		}
	}
}

void normalize_weights(SampleRange * sample_range) {
	auto partitioned_ranges = get_partitioned_ranges(sample_range);
	for (auto label_range : partitioned_ranges) {
		normalize_weights_for_single_label(&label_range);
	}
}

void initialize_predictions(SampleRange * sample_range) {
	for (auto & sample : *sample_range) {
		sample.set_prediction(0.0);
	}
}

void update_predictions(SampleRange * sample_range, DenseTree const & tree) {
	for (auto & sample : *sample_range) {
		sample.set_prediction(sample.get_prediction() + tree.predict(sample.get_pixels()));
	}
}

void set_weights_from_predictions(SampleRange * sample_range, double max_weight) {
	for (auto & sample : *sample_range) {
		double const weight = std::exp(-sample.get_signed_label() * sample.get_prediction());
		double const clamped_weight = std::min(weight, max_weight);
		sample.set_weight(clamped_weight);
	}
	normalize_weights(sample_range);
}

SampleRange get_trimmed_range(SampleRange * sample_range, double trim_fraction) {
	assert(0 <= trim_fraction && trim_fraction < 1.0);

	sort_highest_weight_first(sample_range);

	std::vector<double> cumsum;
	cumsum.reserve(1 + sample_range->size());
	cumsum.push_back(0.0);
	for (auto const & sample : *sample_range) {
		cumsum.push_back(cumsum.back() + sample.get_weight());
	}

	double const subset_weight = (1.0 - trim_fraction) * cumsum.back();
	int trimmed_size = sample_range->size();
	for (int size = 1; size < sample_range->size(); ++size) {
		if (cumsum[size] > subset_weight) {
			trimmed_size = size;
			break;
		}
	}

	return SampleRange(sample_range->begin(), sample_range->begin() + trimmed_size);
}

auto get_trimmed_ranges(SampleRange * sample_range, double trim_fraction) {
	auto partitioned_ranges = get_partitioned_ranges(sample_range);
	decltype(partitioned_ranges) trimmed_ranges;
	trimmed_ranges[POSITIVE] = get_trimmed_range(&partitioned_ranges[POSITIVE], trim_fraction);
	trimmed_ranges[NEGATIVE] = get_trimmed_range(&partitioned_ranges[NEGATIVE], trim_fraction);
	return trimmed_ranges;
}

SampleRange move_sub_ranges_to_front(SampleRange * full_range, std::array<SampleRange, N_LABELS> * sub_ranges) {
	decltype(*sub_ranges) sorted_sub_ranges = *sub_ranges;
	std::sort(sorted_sub_ranges.begin(), sorted_sub_ranges.end(),
		[](SampleRange const & lhs, SampleRange const & rhs) {
		return (lhs.begin() < rhs.begin());
	});
	auto target_begin = full_range->begin();
	for (int sub_idx = 0; sub_idx < N_LABELS; ++sub_idx) {
		auto current_sub_range = sorted_sub_ranges[sub_idx];
		for (int i = 0; i < current_sub_range.size(); ++i) {
			std::swap(target_begin[i], current_sub_range[i]);
		}
		target_begin += current_sub_range.size();
	}
	return SampleRange(full_range->begin(), target_begin);
}

double compute_loss(SampleRange const & sample_range) {
	std::array<double, N_LABELS> sums = { 0.0 };
	std::array<int, N_LABELS> counts = { 0 };
	for (auto const & sample : sample_range) {
		int const label = sample.get_label();
		sums[label] += std::exp(-sample.get_signed_label() * sample.get_prediction());
		counts[label] += 1;
	}

	if (counts[POSITIVE] == 0 || counts[NEGATIVE] == 0) {
		assert(false);
		return 1.0;
	}

	double loss = 0.0;
	for (int label = 0; label < N_LABELS; ++label) {
		loss += sums[label] / counts[label];
	}
	return loss / N_LABELS;
}

int get_label_from_prediction(double prediction) {
	return (prediction > 0.0 ? POSITIVE : NEGATIVE);
}

auto compute_error_rates(SampleRange const & sample_range) {
	std::array<int, N_LABELS> counts = { 0 };
	std::array<int, N_LABELS> error_counts = { 0 };
	for (auto const & sample : sample_range) {
		int const actual_label = sample.get_label();
		int const predicted_label = get_label_from_prediction(sample.get_prediction());
		counts[actual_label] += 1;
		if (predicted_label != actual_label) {
			error_counts[actual_label] += 1;
		}
	}
	std::array<double, N_LABELS> error_rates;
	for (int label = 0; label < N_LABELS; ++label) {
		error_rates[label] =
			(counts[label] > 0 ? static_cast<double>(error_counts[label]) / counts[label] : 1.0);
	}
	return error_rates;
}

auto learn_gab(SampleRange * sample_range) {
	static int const tree_depth = 2;
	static int const min_leaf_occupancy = 0;
	static double const max_weight = 100;
	static int const min_sample_count = 100;
	static double const trim_fraction = 0.05;

	initialize_predictions(sample_range);
	set_weights_from_predictions(sample_range, max_weight);
	double previousLoss = compute_loss(*sample_range);

	std::vector<DenseTree> trees;
	for (int iteration = 0; iteration < 5; ++iteration) {
		//std::cout << "tpr = " << (1.0 - compute_error_rates(*sample_range)[POSITIVE]) << "\n";
		//std::cout << "fpr = " << compute_error_rates(*sample_range)[NEGATIVE] << "\n";
		auto trimmed_ranges = get_trimmed_ranges(sample_range, trim_fraction);
		int const lowest_sample_count = std::min(trimmed_ranges[POSITIVE].size(), trimmed_ranges[NEGATIVE].size());
		if (lowest_sample_count < min_sample_count) {
			std::cout << "Warning: Insufficent samples to continue GAB learning.\n";
			break;
		}

		auto active_range = move_sub_ranges_to_front(sample_range, &trimmed_ranges);
		DenseTree const tree = fit_tree(&active_range, tree_depth, min_leaf_occupancy);
		update_predictions(sample_range, tree);
		set_weights_from_predictions(sample_range, max_weight);
		trees.push_back(tree);

		double const currentLoss = compute_loss(*sample_range);
		double const relativeLossDecrease = (previousLoss - currentLoss) / previousLoss;
		previousLoss = currentLoss;
		std::cout << "loss decrease = " << 100.0 * relativeLossDecrease << " %\n";
	}
	return trees;
}