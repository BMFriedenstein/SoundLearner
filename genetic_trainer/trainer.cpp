/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * trainer.cpp
 *  Created on: 04 Jan 2019
 *      Author: Brandon
 */

#include "genetic_trainer/trainer.h"

#include <stddef.h>
#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <utility>

#include "include/common.h"
#include "include/filereader.h"
#include "include/filewriter.h"

namespace instrument {
inline bool cmp_by_name(const std::unique_ptr<InstrumentModelC>& a, const std::unique_ptr<InstrumentModelC>& b) {
  return a->error_score < b->error_score;
}
}  // namespace instrument

namespace trainer {
InstumentTrainerC::InstumentTrainerC(uint16_t num_starting_occilators,
                                     uint16_t class_size,
                                     const std::vector<int16_t>& audio,
                                     const std::string& location)
    : progress_location(location), src_audio(audio) {
  for (uint16_t i = 0; i < class_size; i++) {
    std::unique_ptr<instrument::InstrumentModelC> new_instrument(
        new instrument::InstrumentModelC(num_starting_occilators, "instrument_" + std::to_string(i)));
    trainees.push_back(std::move(new_instrument));
  }
}

GeneticInstumentTrainerC::GeneticInstumentTrainerC(uint16_t num_starting_occilators,
                                                   uint16_t class_size,
                                                   const std::vector<int16_t>& src_audio,
                                                   const std::string& progress_location,
                                                   uint32_t gens_per_addition)
    : trainer::InstumentTrainerC(num_starting_occilators, class_size, src_audio, progress_location),
      gens_per_addition(gens_per_addition) {}

/*
 * Calculate the average mean error for the generated signals of each trainee instrument.
 * in class
 */
double InstumentTrainerC::GetError(const std::vector<int16_t>& tgt_audio,
                                   double* corr_score,
                                   double* mae_score,
                                   double* diff_score) {
  // Check the src audio is the right size.
  if (tgt_audio.size() != src_audio.size()) {
    std::cout << "WARN !!! BAD audio size" << std::endl;
    return 1.0;
  }

  // Get average energy difference.
  double tgt_energy = 0;
  for (size_t s = 0; s < src_audio.size(); s++) {
    tgt_energy += std::abs(tgt_audio[s]);
  }
  tgt_energy = tgt_energy / src_audio.size();
  *diff_score = std::abs(tgt_energy - src_energy) / static_cast<double>(MAX_AMP);
  if (src_energy / tgt_energy > 5) {
    *diff_score = 1.0;
    *mae_score = 1.0;
    *corr_score = 1.0;
    return 1.0;
  }

  // Get The maen absolute error
  *mae_score = MeanAbsoluteError(tgt_audio, src_energy / tgt_energy);

  // Determine cross correlation
  *corr_score = CrossCorrelation(tgt_audio);
  return *corr_score / 2 + *diff_score / 2 + *mae_score / 400;
}

double InstumentTrainerC::CrossCorrelation(const std::vector<int16_t>& tgt_audio) {
  // Get next power of 2
  auto signal_length = static_cast<uint32_t>(pow(2, ceil(log(src_audio.size() * 2) / log(2))));

  // Pad signals with { 0.0 0.0j }
  std::vector<std::complex<double>> cmplx_src(signal_length, {0.0, 0.0});
  std::vector<std::complex<double>> cmplx_tgt(signal_length, {0.0, 0.0});
  std::vector<std::complex<double>> cmplx_corr(signal_length, {0.0, 0.0});
  for (size_t s = 0; s < src_audio.size(); s++) {
    cmplx_src[s] = src_audio[s];
    cmplx_tgt[src_audio.size() - (1 + s)] = tgt_audio[s];
  }

  //  Fft::convolve(cmplx_src, cmplx_tgt, cmplx_corr);

  return MAX_AMP * src_audio.size() / (std::abs(cmplx_corr[src_audio.size()]));
}

double InstumentTrainerC::MeanAbsoluteError(const std::vector<int16_t>& tgt_audio, const double corr_factor) {
  double abs_error = 0.0;

  for (size_t s = 0; s < src_audio.size(); s++) {
    abs_error += std::abs(corr_factor * tgt_audio[s] - src_audio[s]);
  }

  double mean_abs_error = abs_error / (src_audio.size() * 2 * MAX_AMP);

  return mean_abs_error;
}

/*
 * Calculate the average mean error for the generated
 * signals of each trainee instrument in the class.
 */
void GeneticInstumentTrainerC::DetermineFitness() {
  bool have_new_best = false;
  double ave_error = 0.0;
  double min_error = std::numeric_limits<double>::max();
  std::vector<int16_t> best_instrument_sample(src_audio.size());
  std::string best_instrument_json("");
  uint32_t best_index = 0;

  // Determine error score for each instrument for class.
  std::vector<int16_t> temp_sample(src_audio.size());
  for (size_t i = 0; i < trainees.size(); i++) {
    // Only re-calculate score if we have not already done so.
    if (!trainees[i]->score_is_cached) {
      bool has_distorted;
      temp_sample = trainees[i]->GenerateIntSignal(velocity, base_frequency, src_audio.size(), has_distorted);
      trainees[i]->error_score =
          GetError(temp_sample, &trainees[i]->corr_score, &trainees[i]->mae_score, &trainees[i]->diff_score);
      trainees[i]->score_is_cached = true;
    }

    // Add score to average if it is not infinite
    if (trainees[i]->error_score < std::numeric_limits<double>::max()) {
      ave_error += trainees[i]->error_score;

      if (trainees[i]->error_score < min_error) {
        min_error = trainees[i]->error_score;
        best_index = i;

        // Determine best score and if it is a new best score
        if (best_index > 0) {
          have_new_best = true;
          best_instrument_json = trainees[i]->ToJson();
          best_instrument_sample = temp_sample;
        }
      }
    }
  }
  ave_error = ave_error / trainees.size();

  // Log progression and write out best sample if it is not new.
  if (!progress_location.empty()) {
    std::cout << std::to_string(gen_count) + ", " + std::to_string(min_error) + ", " +
                     std::to_string(trainees[best_index]->corr_score) + ", " +
                     std::to_string(trainees[best_index]->mae_score) + ", " +
                     std::to_string(trainees[best_index]->diff_score) + ", " + std::to_string(ave_error);

    if (have_new_best) {
      filewriter::text::WriteFile(progress_location + "/Gen_" + std::to_string(gen_count) + ".json",
                                  best_instrument_json);
      filewriter::wave::MonoWaveWriterC wave_writer(best_instrument_sample);
      wave_writer.Write(progress_location + "/Gen_" + std::to_string(gen_count) + ".wav");
    }
  }
}

/*
 * Determine which trainees survived this generation,
 * Replace killed off instruments with mutated survivors.
 */
void GeneticInstumentTrainerC::GeneticAlgorithm() {
  // Sort trainees. Then replace the bottom 75% trainees.
  std::sort(trainees.begin(), trainees.end(), instrument::cmp_by_name);
  size_t keep_amount = trainees.size() / 3;
  for (size_t i = 0; i < trainees.size(); i++) {
    size_t keep_index = i % (keep_amount);
    if (i >= keep_amount) {
      trainees[i].reset(trainees[keep_index]->TuneInstrument(40).release());
    }
  }
}

void GeneticInstumentTrainerC::Start(uint16_t a_num_of_generations) {
  num_generations = a_num_of_generations;

  // Pre-calculate energy of src wave
  src_energy = 0.0;
  for (size_t s = 0; s < src_audio.size(); s++) {
    src_energy += std::abs(src_audio[s]);
  }
  src_energy = src_energy / src_audio.size();
  if (src_energy == 0) {
    exit(EXIT_BAD_SOURCE_SIGNAL);
  }
  std::cout << "src average energy: " << src_energy << std::endl;
  std::cout << "Generation, Error, Correlation, MAE, Average difference, Average error";
  for (gen_count = 0; gen_count < num_generations; gen_count++) {
    DetermineFitness();
    GeneticAlgorithm();

    // Add additional oscillator.
    if (gen_count != 0 && gens_per_addition > 0 && gen_count % gens_per_addition == 0) {
      for (size_t i = 0; i < trainees.size(); i++) {
        trainees[i]->score_is_cached = false;
        trainees[i]->error_score = std::numeric_limits<double>::max();
        trainees[i]->AddUntunedString();
      }
    }
  }
}
}  // namespace trainer
