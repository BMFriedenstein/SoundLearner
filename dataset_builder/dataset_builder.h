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
#include <random>
#include <utility>
#include <vector>

#include "include/common.h"

class DataBuilder {
private:
  static constexpr char data_output[] = "data";
  // Define the range.
  std::size_t num_samples;
  std::size_t min_coupled_oscilators;
  std::size_t max_coupled_oscilators;
  std::size_t min_uncoupled_oscilators;
  std::size_t max_uncoupled_oscilators;
  std::size_t starting_index;
  double min_note_frequency;
  double max_note_frequency;
  double min_frequency_factor;
  double max_frequency_factor;
  bool require_fundamental;
  std::vector<double> coupled_frequency_factors;
  std::mt19937 rand_eng;

public:
  void DataBuildJob(std::size_t index);

  DataBuilder(std::size_t sample_time_secs, std::size_t min_coupled_count, std::size_t max_coupled_count,
              std::size_t min_uncoupled_count = 0, std::size_t max_uncoupled_count = 0, std::size_t first_index = 0,
              double min_note_freq = 1000.0, double max_note_freq = 1000.0, double min_freq_factor = 0.0,
              double max_freq_factor = 1.0, bool require_fundamental_oscillator = false,
              std::vector<double> coupled_freq_factors = {}, std::size_t rand_seed = std::random_device{}())
      : num_samples(SAMPLE_RATE * sample_time_secs), min_coupled_oscilators(min_coupled_count),
        max_coupled_oscilators(std::max(min_coupled_count, max_coupled_count)), min_uncoupled_oscilators(min_uncoupled_count),
        max_uncoupled_oscilators(std::max(min_uncoupled_count, max_uncoupled_count)), starting_index(first_index),
        min_note_frequency(std::min(min_note_freq, max_note_freq)), max_note_frequency(std::max(min_note_freq, max_note_freq)),
        min_frequency_factor(std::min(min_freq_factor, max_freq_factor)), max_frequency_factor(std::max(min_freq_factor, max_freq_factor)),
        require_fundamental(require_fundamental_oscillator), coupled_frequency_factors(std::move(coupled_freq_factors)),
        rand_eng(static_cast<std::mt19937::result_type>(rand_seed)) {}
};
#endif // DATASET_BUILDER_H_
