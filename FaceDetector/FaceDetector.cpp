#include "stdafx.h"  

#include "Training.h"

#include "gtest/gtest.h"

#include <utility>
#include <random>

TEST(training, sample) {
	Sample sample(POSITIVE, { 1, 2, 4 });
	EXPECT_EQ(1, sample.get_label());
	EXPECT_EQ(2, sample.get_pixels()[1]);
}

TEST(training, pair_index) {
	EXPECT_EQ(0, pair_to_linear({ 0, 1 }));
	EXPECT_EQ(2, pair_to_linear({ 1, 2 }));

	EXPECT_EQ(std::make_pair(0, 1), linear_to_pair(0));
	EXPECT_EQ(std::make_pair(1, 2), linear_to_pair(2));

	static int const MAX_IDX = 400;
	for (int m = 0; m < MAX_IDX; ++m) {
		for (int n = 0; n < m; ++n) {
			auto pair = std::make_pair(n, m);
			int linear = pair_to_linear(pair);
			ASSERT_EQ(pair, linear_to_pair(linear)) << "linear = " << linear;
		}
	}
}

TEST(training, linear_to_pair_gpu) {
	int const index_count = 400;

	std::vector<std::pair<int, int>> pairs_cpu;
	for (int i = 0; i < index_count; ++i) {
		pairs_cpu.push_back(linear_to_pair(i));
	}

	std::array<int, index_count> n_gpu;
	std::array<int, index_count> m_gpu;
	concurrency::array_view<int> n_gpu_view(n_gpu);
	concurrency::array_view<int> m_gpu_view(m_gpu);
	concurrency::parallel_for_each(
		n_gpu_view.extent,
		[=](concurrency::index<1> i) restrict(amp) {
		linear_to_pair(&n_gpu_view[i], &m_gpu_view[i], i[0]);
	});
	n_gpu_view.synchronize();
	m_gpu_view.synchronize();
	for (int i = 0; i < index_count; ++i) {
		auto pair_gpu = std::make_pair(n_gpu[i], m_gpu[i]);
		ASSERT_EQ(pairs_cpu[i], pair_gpu) << "for linear index = " << i;
	}
}

TEST(training, normalized_pixel_difference_gpu) {
	auto npd_table_cpu = compute_npd_table();

	std::array<unsigned int, 256 * 256> npd_table_gpu;
	concurrency::array_view<unsigned int, 2> npd_table_gpu_view(256, 256, npd_table_gpu.data());
	npd_table_gpu_view.discard_data();
	concurrency::parallel_for_each(
		npd_table_gpu_view.extent,
		[=](concurrency::index<2> index) restrict(amp) {
		npd_table_gpu_view[index] = get_npd(index[0], index[1]);
	});
	npd_table_gpu_view.synchronize();
	for (int i = 0; i < 256; ++i) {
		for (int j = 0; j < 256; ++j) {
			EXPECT_EQ(npd_table_cpu[i][j], npd_table_gpu[i * 256 + j]) <<
				"i = " << i << " j = " << j;
		}
	}
}

TEST(training, split) {
	EXPECT_TRUE(goes_left(17, 18));
	EXPECT_TRUE(goes_left(18, 18));
	EXPECT_FALSE(goes_left(19, 18));
}

TEST(training, sort_lowest_weight_first) {
	std::vector<Sample> samples = {
		Sample(NEGATIVE, { }, 2.0),
		Sample(NEGATIVE, { }, 1.0) };
	sort_lowest_weight_first(&SampleRange(samples));
	EXPECT_LT(samples[0].get_weight(), samples[1].get_weight());
}

TEST(training, sort_highest_weight_first) {
	std::vector<Sample> samples = {
		Sample(NEGATIVE, { }, 2.0),
		Sample(NEGATIVE, { }, 1.0) };
	sort_highest_weight_first(&SampleRange(samples));
	EXPECT_GT(samples[0].get_weight(), samples[1].get_weight());
}

TEST(training, sort_positives_first_and_get_midpoint) {
	std::vector<Sample> samples = {
		Sample(NEGATIVE, { }),
		Sample(NEGATIVE, { }),
		Sample(POSITIVE, { }) };
	auto midpoint = sort_positives_first_and_get_midpoint(&SampleRange(samples));
	EXPECT_EQ(POSITIVE, samples[0].get_label());
	EXPECT_EQ(NEGATIVE, samples[1].get_label());
	EXPECT_EQ(&samples[1], midpoint);
}

TEST(training, get_weighted_mean) {
	std::vector<Sample> samples = {
		Sample(NEGATIVE, { }, 1.0),
		Sample(NEGATIVE, { }, 2.0),
		Sample(POSITIVE, { }, 4.0) };
	auto weighted_mean = get_weighted_mean(SampleRange(samples));
	EXPECT_DOUBLE_EQ((-1.0 * 3.0 + 1.0 * 4.0) / 7.0, weighted_mean);
}

TEST(training, accumulate_histograms) {
	std::vector<Sample> samples = {
		Sample(POSITIVE, { 2, 2, 0 }, 1.0),
		Sample(POSITIVE, { 2, 2, 255 }, 3.0) };

	auto histograms = accumulate_histograms_cpu(&SampleRange(samples));

	EXPECT_DOUBLE_EQ(1.0, histograms[pair_to_linear({ 1, 2 })][g_npd_table[2][0]]);
	EXPECT_DOUBLE_EQ(4.0, histograms[pair_to_linear({ 0, 1 })][g_npd_table[2][2]]);
}

auto get_empty_histograms() {
	histograms_type histograms;
	for (int label = 0; label < N_LABELS; ++label) {
		histograms[label].emplace_back();
		histograms[label].back().fill(0.0);
	}
	return histograms;
}

TEST(training, compute_weight_totals) {
	auto histogram = get_empty_histograms();
	histogram[POSITIVE].front()[0] = 1.0;
	histogram[POSITIVE].front()[1] = 2.0;
	histogram[NEGATIVE].front()[0] = 4.0;

	auto weight_totals = compute_weight_totals(histogram);

	EXPECT_DOUBLE_EQ(3.0, weight_totals[POSITIVE]);
	EXPECT_DOUBLE_EQ(4.0, weight_totals[NEGATIVE]);
}

TEST(training, compute_best_split_no_split) {
	auto histograms = get_empty_histograms();
	auto split = compute_best_split({ histograms });
	EXPECT_EQ(Split::Dummy().get_pixel_indices(), split.get_pixel_indices());
	EXPECT_EQ(Split::Dummy().get_threshold(), split.get_threshold());
}

TEST(training, compute_best_split_no_good_split) {
	auto histograms = get_empty_histograms();
	histograms[NEGATIVE][0][0] = 1.0;
	histograms[POSITIVE][0][0] = 1.0;
	auto split = compute_best_split({ histograms });
	EXPECT_EQ(Split::Dummy().get_pixel_indices(), split.get_pixel_indices());
	EXPECT_EQ(Split::Dummy().get_threshold(), split.get_threshold());
}

TEST(training, compute_best_split_low_split) {
	auto histograms = get_empty_histograms();
	histograms[NEGATIVE][0][0] = 1.0;
	histograms[POSITIVE][0][1] = 1.0;
	auto split = compute_best_split({ histograms });
	EXPECT_EQ(0, pair_to_linear(split.get_pixel_indices()));
	EXPECT_EQ(0, split.get_threshold());
}

TEST(training, compute_best_split_high_split) {
	auto histograms = get_empty_histograms();
	histograms[NEGATIVE][0][254] = 1.0;
	histograms[POSITIVE][0][255] = 1.0;
	auto split = compute_best_split({ histograms });
	EXPECT_EQ(0, pair_to_linear(split.get_pixel_indices()));
	EXPECT_EQ(254, split.get_threshold());
}

TEST(training, compute_best_split_multiple_features) {
	histograms_type histograms;
	for (int label = 0; label < N_LABELS; ++label) {
		for (int i = 0; i < 2; ++i) {
			histograms[label].emplace_back();
			histograms[label].back().fill(0.0);
		}
	}

	// Feature 0: Does not split the data.
	histograms[NEGATIVE][0][17] = 1.0;
	histograms[POSITIVE][0][17] = 1.0;

	// Feature 1: Splits the data.
	histograms[NEGATIVE][1][17] = 1.0;
	histograms[POSITIVE][1][18] = 1.0;

	auto split = compute_best_split(histograms);
	EXPECT_EQ(1, pair_to_linear(split.get_pixel_indices()));
	EXPECT_EQ(17, split.get_threshold());
}

TEST(training, get_split_count) {
	EXPECT_EQ(1, get_split_count(1));
	EXPECT_EQ(3, get_split_count(2));
	EXPECT_EQ(7, get_split_count(3));
	EXPECT_EQ(15, get_split_count(4));
}

TEST(training, get_partitioned_ranges) {
	std::vector<Sample> samples;
	int counter = 0;
	std::generate_n(std::back_inserter(samples), 100, [&]() {
		counter += 1;
		int label = counter % N_LABELS;
		return Sample(label, { });
	});

	auto partitioned_ranges = get_partitioned_ranges(&SampleRange(samples));

	for (int label = 0; label < N_LABELS; ++label) {
		for (auto const & sample : partitioned_ranges[label]) {
			ASSERT_EQ(label, sample.get_label()) << "for label " << label;
		}
	}
}

TEST(training, fit_tree) {
	std::vector<Sample> samples = {
		Sample(NEGATIVE, { 0, 1, 2 }),
		Sample(NEGATIVE, { 0, 2, 1 }),
		Sample(NEGATIVE, { 1, 0, 2 }),
		Sample(NEGATIVE, { 1, 2, 0 }),
		Sample(NEGATIVE, { 2, 1, 0 }),
		Sample(POSITIVE, { 2, 0, 1 }) };

	auto const tree = fit_tree(&SampleRange(samples), 2, 0);

	ASSERT_TRUE(tree.is_complete());
	EXPECT_EQ(2, tree.get_depth());
	for (auto const & sample : samples) {
		auto label = sample.get_label();
		auto prediction = tree.predict(sample.get_pixels());
		if (label == POSITIVE) {
			EXPECT_DOUBLE_EQ(1.0, prediction);
		}
		else if (label == NEGATIVE) {
			EXPECT_DOUBLE_EQ(-1.0, prediction);
		}
		else {
			FAIL();
		}
	}
}

TEST(training, clamp) {
	EXPECT_EQ(0, clamp(-1, 0, 4));
	EXPECT_EQ(2, clamp(2, 0, 4));
	EXPECT_EQ(4, clamp(8, 0, 4));
}

TEST(training, normalize_weights_for_single_label) {
	// Empty.
	std::vector<Sample> samples;
	normalize_weights_for_single_label(&SampleRange(samples));

	// All zero.
	std::generate_n(std::back_inserter(samples), 10, []() {
		return Sample(POSITIVE, { }, 0.0);
	});
	normalize_weights_for_single_label(&SampleRange(samples));
	for (auto const & sample : samples) {
		ASSERT_DOUBLE_EQ(1.0 / 10.0, sample.get_weight());
	}

	// Some non-zero.
	std::generate_n(std::back_inserter(samples), 10, []() {
		return Sample(POSITIVE, { }, 5.0);
	});
	normalize_weights_for_single_label(&SampleRange(samples));
	double sum = 0.0;
	for (auto const & sample : samples) {
		sum += sample.get_weight();
	}
	EXPECT_DOUBLE_EQ(1.0, sum);
}

TEST(training, get_trimmed_range) {
	std::vector<Sample> samples = {
		Sample(NEGATIVE, { }, 1.0),
		Sample(NEGATIVE, { }, 2.0),
		Sample(NEGATIVE, { }, 3.0),
		Sample(NEGATIVE, { }, 4.0),
		Sample(NEGATIVE, { }, 5.0) };

	double trim_fraction = 6.5 / 15.0;
	auto trimmed_range = get_trimmed_range(&SampleRange(samples), trim_fraction);
	ASSERT_EQ(2, trimmed_range.size());
	EXPECT_EQ(5.0, trimmed_range[0].get_weight());
	EXPECT_EQ(4.0, trimmed_range[1].get_weight());
}

TEST(training, move_sub_ranges_to_front) {
	std::vector<Sample> samples = {
		Sample(NEGATIVE,{}, 1.0),
		Sample(NEGATIVE,{}, 2.0),
		Sample(NEGATIVE,{}, 3.0),
		Sample(NEGATIVE,{}, 4.0),
		Sample(NEGATIVE,{}, 5.0) };

	auto full_range = SampleRange(samples);
	std::array<SampleRange, N_LABELS> sub_ranges = {
		SampleRange(&samples[3], &samples[4] + 1),
		SampleRange(&samples[1], &samples[1] + 1) };

	auto compacted_range = move_sub_ranges_to_front(&full_range, &sub_ranges);
	ASSERT_EQ(3, compacted_range.size());
	EXPECT_EQ(2.0, compacted_range[0].get_weight());
	EXPECT_EQ(4.0, compacted_range[1].get_weight());
	EXPECT_EQ(5.0, compacted_range[2].get_weight());
}

class RandomSampleGenerator {
public:
	RandomSampleGenerator(int sample_count, int pixel_count) {
		_samples.reserve(sample_count);
		std::default_random_engine generator(42);
		std::uniform_int_distribution<int> pixel_distribution(0, 255);
		std::uniform_real_distribution<double> weight_distribution(0.0, 1.0);
		std::uniform_int_distribution<int> label_distribution(0, 1);
		for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
			std::vector<unsigned char> pixels(pixel_count);
			std::generate(pixels.begin(), pixels.end(), [&]() {
				return pixel_distribution(generator);
			});
			int label = label_distribution(generator);
			double weight = weight_distribution(generator);
			_samples.emplace_back(label, std::move(pixels), weight);
		}
	}

	auto get_samples() { return SampleRange(_samples); }
private:
	std::vector<Sample> _samples;
};

TEST(training, accumulate_histograms_comparison) {
	RandomSampleGenerator data(101, 400);

	auto res_gpu = accumulate_histograms_gpu(&data.get_samples());
	auto res_cpu = accumulate_histograms_cpu(&data.get_samples());

	for (int i = 0; i < static_cast<int>(res_cpu.size()); ++i) {
		for (int j = 0; j < 256; j++) {
			ASSERT_NEAR(res_cpu[i][j], res_gpu[i][j], 1e-3) <<
				"[" << i << ", " << j << "]";
		}
	}
}

// Benchmarking.
RandomSampleGenerator g_data(101, 400);
TEST(training, accumulate_histograms_cpu) {
	auto res = accumulate_histograms_cpu(&g_data.get_samples());
	ASSERT_FALSE(res.empty());
}
TEST(training, accumulate_histograms_gpu) {
	auto res = accumulate_histograms_gpu(&g_data.get_samples());
	ASSERT_FALSE(res.empty());
}

static double const PI = 3.1415;

class WaveSampleGenerator {
public:
	WaveSampleGenerator(int sample_count, int pixel_count) {
		double const sin_magnitude = 256 / 8;
		double const sin_period = 25;
		int const noise_magnitude = 256 / 4;
		int const noise_mean = 256 / 2;
		std::default_random_engine generator(42);
		std::uniform_int_distribution<int> noise_distribution(
			noise_mean - noise_magnitude,
			noise_mean + noise_magnitude);
		std::uniform_real_distribution<double> phase_distribution(0, 2 * PI);

		_samples.reserve(sample_count);
		for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
			std::vector<unsigned char> pixels(pixel_count);
			int label = (sample_idx % 20 == 0 ? POSITIVE : NEGATIVE);
			if (label == POSITIVE) {
				for (int i = 0; i < pixel_count; ++i) {
					double const sin_phase = phase_distribution(generator);
					int const sin_value = static_cast<int>(
						sin_magnitude * std::sin(2 * PI * i / sin_period + sin_phase));
					pixels[i] = clamp(noise_distribution(generator) + sin_value, 0, 255);
				}
			}
			else {
				std::generate(pixels.begin(), pixels.end(), [&]() {
					return clamp(noise_distribution(generator), 0, 255);
				});
			}
			_samples.emplace_back(label, std::move(pixels));
		}
	}

	auto get_training_samples() {
		return SampleRange(_samples.data(), _samples.data() + _samples.size() / 2);
	}

	auto get_test_samples() {
		return SampleRange(_samples.data() + _samples.size() / 2, _samples.data() + _samples.size());
	}

private:
	std::vector<Sample> _samples;
};

TEST(learning, learn_gab) {
	auto data = WaveSampleGenerator(100000, 400);
	auto const trees = learn_gab(&data.get_training_samples());

	auto test_samples = data.get_test_samples();
	for (auto & sample : test_samples) {
		double prediction = 0.0;
		for (auto const & tree : trees) {
			prediction += tree.predict(sample.get_pixels());
		}
		sample.set_prediction(prediction);
	}
	auto const error_rates = compute_error_rates(test_samples);
	std::cout << "true positive rate = " << (1.0 - error_rates[POSITIVE]) << "\n";
	std::cout << "false positive rate = " << error_rates[NEGATIVE] << "\n";
}

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
	//std::getchar(); // keep console window open until Return keystroke
}