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
  std::size_t coupled_oscilators;
  std::size_t uncoupled_oscilators;
  std::size_t starting_index;
  std::size_t img_resolution;
  bool write_ppm_previews;

public:
  void DataBuildJob(double velocity, double freq, std::size_t index);

  DataBuilder(std::size_t sample_time_secs, std::size_t coupled_oscilators, std::size_t uncoupled_oscilators = 0, std::size_t starting_index = 0,
              std::size_t img_resolution = 512U, bool write_ppm_previews = false, std::size_t rand_seed = std::random_device{}())
      : rand_eng(rand_seed), num_samples(SAMPLE_RATE * sample_time_secs), coupled_oscilators(coupled_oscilators),
        uncoupled_oscilators(uncoupled_oscilators), starting_index(starting_index), img_resolution(img_resolution), write_ppm_previews(write_ppm_previews) {}
};
#endif // DATASET_BUILDER_H_
