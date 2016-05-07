#include "stdafx.h"  

#include "Training.h"
#include <iostream>
#include <utility>

#include "gtest/gtest.h"

TEST(training, sample) {
	Sample sample(POSITIVE, { 1, 2, 4 });
	EXPECT_EQ(1, sample.get_label());
	EXPECT_EQ(2, sample.get_features()[1]);

	Sample otherSample(NEGATIVE, { 8, 16 });
	swap(sample, otherSample);
	EXPECT_EQ(NEGATIVE, sample.get_label());
	EXPECT_EQ(16, sample.get_features()[1]);
	EXPECT_EQ(POSITIVE, otherSample.get_label());
	EXPECT_EQ(2, otherSample.get_features()[1]);
}

TEST(training, pair_index) {
	EXPECT_EQ(0, pair_to_linear(std::make_pair(0, 1)));
	EXPECT_EQ(2, pair_to_linear(std::make_pair(1, 2)));

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

TEST(training, accumulate_histograms) {
	std::vector<Sample> samples;
	samples.push_back(Sample(NEGATIVE, { 0, 1 }, 4));
	samples.push_back(Sample(POSITIVE, { 2, 2 }, 6));
	samples.push_back(Sample(POSITIVE, { 1, 2 }, 8));

	auto histograms = accumulate_histograms(SampleRange(samples));

	EXPECT_DOUBLE_EQ(4, histograms[1][NEGATIVE][1]); // feature 1, bin 1
	EXPECT_DOUBLE_EQ(6, histograms[0][POSITIVE][2]); // feature 0, bin 2
	EXPECT_DOUBLE_EQ(6 + 8, histograms[1][POSITIVE][2]); // feature 1, bin 2

	auto weight_sums = compute_weight_totals(histograms.front());

	EXPECT_DOUBLE_EQ(4, weight_sums[NEGATIVE]);
	EXPECT_DOUBLE_EQ(6 + 8, weight_sums[POSITIVE]);
}

histogram_type get_empty_histogram() {
	histogram_type histogram;
	histogram[NEGATIVE].fill(0.0);
	histogram[POSITIVE].fill(0.0);
	return histogram;
}

TEST(training, compute_best_split_no_split) {
	auto histogram = get_empty_histogram();
	auto split = compute_best_split({ histogram });
	EXPECT_EQ(0, split.get_feature_index());
	EXPECT_EQ(0, split.get_threshold());
}

TEST(training, compute_best_split_no_good_split) {
	auto histogram = get_empty_histogram();
	histogram[NEGATIVE][0] = 1.0;
	histogram[POSITIVE][0] = 1.0;
	auto split = compute_best_split({ histogram });
	EXPECT_EQ(0, split.get_feature_index());
	EXPECT_EQ(0, split.get_threshold());
}

TEST(training, compute_best_split_low_split) {
	auto histogram = get_empty_histogram();
	histogram[NEGATIVE][0] = 1.0;
	histogram[POSITIVE][1] = 1.0;
	auto split = compute_best_split({ histogram });
	EXPECT_EQ(0, split.get_feature_index());
	EXPECT_EQ(0, split.get_threshold());
}

TEST(training, compute_best_split_high_split) {
	auto histogram = get_empty_histogram();
	histogram[NEGATIVE][254] = 1.0;
	histogram[POSITIVE][255] = 1.0;
	auto split = compute_best_split({ histogram });
	EXPECT_EQ(254, split.get_threshold());
}

TEST(training, compute_best_split_multiple_features) {
	// Does not split the data.
	auto bad_histogram = get_empty_histogram();
	bad_histogram[NEGATIVE][17] = 1.0;
	bad_histogram[POSITIVE][17] = 1.0;

	// Splits the data.
	auto good_histogram = get_empty_histogram();
	good_histogram[NEGATIVE][17] = 1.0;
	good_histogram[POSITIVE][18] = 1.0;

	auto split = compute_best_split({ bad_histogram, good_histogram });
	EXPECT_EQ(1, split.get_feature_index());
}

TEST(training, split) {
	Split split(0, 2);
	EXPECT_TRUE(split.goes_left({ 1 }));
	EXPECT_TRUE(split.goes_left({ 2 }));
	EXPECT_FALSE(split.goes_left({ 3 }));
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}