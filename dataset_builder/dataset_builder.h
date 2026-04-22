/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * builder_thread_pool.h
 *  Created on: 13 June 2020
 *      Author: Brandon
 */
#pragma once
#ifndef DATASET_BUILDER_H_
#define DATASET_BUILDER_H_
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <utility>

#include "include/common.h"
#include "include/fft.h"
#include "instrument/instrument_model.h"

class DataBuilder {
private:
  static constexpr char data_output[] = "data";
  std::mt19937 rand_eng;
  // Define the range.
  std::size_t num_samples;
  std::size_t min_coupled_oscilators;
  std::size_t max_coupled_oscilators;
  std::size_t min_uncoupled_oscilators;
  std::size_t max_uncoupled_oscilators;
  std::size_t starting_index;
  std::size_t img_resolution;
  std::size_t frequency_bins;
  std::size_t time_frames;
  std::size_t fft_size_multiplier;
  bool write_ppm_previews;

public:
  void DataBuildJob(double freq, std::size_t index);

  DataBuilder(std::size_t sample_time_secs, std::size_t min_coupled_count, std::size_t max_coupled_count,
              std::size_t min_uncoupled_count = 0, std::size_t max_uncoupled_count = 0, std::size_t first_index = 0,
              std::size_t preview_resolution = 512U, std::size_t feature_frequency_bins = 512U, std::size_t feature_time_frames = 512U,
              std::size_t feature_fft_size_multiplier = 25U, bool should_write_ppm_previews = false,
              std::size_t rand_seed = std::random_device{}())
      : rand_eng(rand_seed), num_samples(SAMPLE_RATE * sample_time_secs), min_coupled_oscilators(min_coupled_count),
        max_coupled_oscilators(std::max(min_coupled_count, max_coupled_count)), min_uncoupled_oscilators(min_uncoupled_count),
        max_uncoupled_oscilators(std::max(min_uncoupled_count, max_uncoupled_count)), starting_index(first_index),
        img_resolution(preview_resolution), frequency_bins(feature_frequency_bins), time_frames(feature_time_frames), fft_size_multiplier(feature_fft_size_multiplier),
        write_ppm_previews(should_write_ppm_previews) {}
};
#endif // DATASET_BUILDER_H_
