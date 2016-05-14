#include "stdafx.h"  

#include "Training.h"
#include <iostream>
#include <utility>

#include "gtest/gtest.h"

TEST(training, sample) {
	Sample sample(POSITIVE, { 1, 2, 4 });
	EXPECT_EQ(1, sample.get_label());
	EXPECT_EQ(2, sample.get_pixels()[1]);

	Sample otherSample(NEGATIVE, { 8, 16 });
	swap(sample, otherSample);
	EXPECT_EQ(NEGATIVE, sample.get_label());
	EXPECT_EQ(16, sample.get_pixels()[1]);
	EXPECT_EQ(POSITIVE, otherSample.get_label());
	EXPECT_EQ(2, otherSample.get_pixels()[1]);
}

TEST(training, pair_index) {
	EXPECT_EQ(0, pair_to_linear({ 0, 1 }));
	EXPECT_EQ(2, pair_to_linear({ 1, 2 }));

	EXPECT_EQ(std::make_pair(0, 1), linear_to_pair(0));
	EXPECT_EQ(std::make_pair(1, 2), linear_to_pair(2));

	static int const MAX_IDX = 256;
	for (int m = 0; m < MAX_IDX; ++m) {
		for (int n = 0; n < m; ++n) {
			auto pair = std::make_pair(n, m);
			int linear = pair_to_linear(pair);
			ASSERT_EQ(pair, linear_to_pair(linear)) << "linear = " << linear;
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
		Sample(NEGATIVE, { }, 2),
		Sample(NEGATIVE, { }, 1) };
	sort_lowest_weight_first(&SampleRange(samples));
	EXPECT_LT(samples[0].get_weight(), samples[1].get_weight());
}


TEST(training, accumulate_histograms) {
	std::vector<Sample> samples = {
		Sample(POSITIVE, { 2, 2, 0 }, 1),
		Sample(POSITIVE, { 2, 2, 255 }, 3) };

	auto histograms = accumulate_histograms_cpu(&SampleRange(samples));

	EXPECT_DOUBLE_EQ(1, histograms[pair_to_linear({ 1, 2 })][g_npd_table[2][0]]);
	EXPECT_DOUBLE_EQ(1 + 3, histograms[pair_to_linear({ 0, 1 })][g_npd_table[2][2]]);
}

auto get_empty_histograms() {
	histograms_type histograms;
	for (int label = 0; label < N_LABELS; ++label) {
		histograms[label].emplace_back();
		histograms[label].back().fill(0.0);
	}
	return histograms;
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

TEST(training, fit_tree) {
	std::vector<Sample> samples;
	samples.push_back(Sample(NEGATIVE, { 0, 1, 2 }));
	samples.push_back(Sample(NEGATIVE, { 0, 2, 1 }));
	samples.push_back(Sample(NEGATIVE, { 1, 0, 2 }));
	samples.push_back(Sample(NEGATIVE, { 1, 2, 0 }));
	samples.push_back(Sample(NEGATIVE, { 2, 1, 0 }));
	samples.push_back(Sample(POSITIVE, { 2, 0, 1 }));

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

class DummySampleGenerator {
public:
	DummySampleGenerator(int sample_count, int pixel_count) {
		_samples.reserve(sample_count);
		srand(42);
		for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
			std::vector<unsigned char> pixels(pixel_count);
			std::generate(pixels.begin(), pixels.end(), []() { return rand() % 256; });
			int label = rand() % 2;
			double weight = static_cast<double>(rand()) / RAND_MAX;
			_samples.emplace_back(label, std::move(pixels), weight);
		}
	}

	auto & get_samples() { return _samples; }
private:
	std::vector<Sample> _samples;
};

TEST(training, normalized_pixel_difference_gpu) {
	auto npd_table_cpu = compute_npd_table();

	std::array<unsigned int, 256 * 256> npd_table_gpu;
	concurrency::array_view<unsigned int, 2> npd_table_gpu_view(256, 256, npd_table_gpu.data());
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

TEST(training, accumulate_histograms_comparison) {
	DummySampleGenerator data(501, 400);

	auto res_gpu = accumulate_histograms_gpu(&SampleRange(data.get_samples()));
	auto res_cpu = accumulate_histograms_cpu(&SampleRange(data.get_samples()));

	for (int i = 0; i < static_cast<int>(res_cpu.size()); ++i) {
		for (int j = 0; j < 256; j++) {
			ASSERT_NEAR(res_cpu[i][j], res_gpu[i][j], 1e-4) <<
				"[" << i << ", " << j << "]";
		}
	}
}

DummySampleGenerator g_data(501, 401);

//TEST(training, accumulate_histograms_cpu) {
//	auto res = accumulate_histograms_cpu(&SampleRange(g_data.get_samples()));
//	ASSERT_FALSE(res.empty());
//}

TEST(training, accumulate_histograms_gpu) {
	auto res = accumulate_histograms_gpu(&SampleRange(g_data.get_samples()));
	ASSERT_FALSE(res.empty());
}

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
	//std::getchar(); // keep console window open until Return keystroke
}