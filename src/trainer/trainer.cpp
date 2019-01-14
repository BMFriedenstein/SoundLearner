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

#include "trainer/trainer.h"

#include <cmath>
#include <cstdio>

#include <fstream>
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
// #include <thread>

#include "wave/wave.h"
#include "fft/fft.h"

namespace instrument {
inline bool cmp_by_name(const std::unique_ptr<InstrumentModelC>& a,
                        const std::unique_ptr<InstrumentModelC>& b) {
  return a->error_score_ < b->error_score_;
}
}

namespace trainer {
InstumentTrainerC::InstumentTrainerC(uint16_t num_starting_occilators,
                                     uint16_t class_size,
                                     std::vector<int16_t>& src_audio,
                                     std::string& progress_location) {
  progress_location_ = progress_location;
  src_audio_ = src_audio;
  trainees_.reserve(class_size);
  for (uint16_t i = 0; i < class_size; i++) {
    std::string name_of_instrument = "instrument_" + std::to_string(i);
    std::unique_ptr<instrument::InstrumentModelC> new_instrument(
        new instrument::InstrumentModelC(num_starting_occilators,
                                         name_of_instrument));
    trainees_.push_back(std::move(new_instrument));
  }
}

GeneticInstumentTrainerC::GeneticInstumentTrainerC(
    uint16_t num_starting_occilators, uint16_t class_size,
    std::vector<int16_t>& src_audio, std::string& progress_location,
    uint32_t gens_per_addition)
    : trainer::InstumentTrainerC(num_starting_occilators, class_size,
                                 src_audio, progress_location),
      gens_per_addition_(gens_per_addition) {
  std::vector<bool> sustain = std::vector<bool>(src_audio.size(), true);
}

/*
 * Calculate the average mean error for the generated signals of each trainee instrument.
 * in class
 */
double InstumentTrainerC::GetError( const std::vector<int16_t>& tgt_audio) {

  // Check the src audio is the right size.
  if (tgt_audio.size() != src_audio_.size()) {
    std::cout << "WARN !!! BAD audio size" << std::endl;
    return 1.0;
  }

  // Get average energy difference.
  double tgt_energy = 0;
  for (size_t s = 0; s < src_audio_.size(); s++) {
    tgt_energy += std::abs(tgt_audio[s]);
  }
  tgt_energy = tgt_energy / src_audio_.size();
  double ave_energy_diff = std::abs(tgt_energy - src_energy_)
      / static_cast<double>(MAX_AMP);
  if (src_energy_ / tgt_energy > 5) {
    return 1.0;
  }

  // Get The maximum difference.
  double src_max = *std::max_element(src_audio_.begin(), src_audio_.end());
  double tgt_max = *std::max_element(tgt_audio.begin(), tgt_audio.end());
  double max_diff = std::abs(tgt_max -src_max)/static_cast<double>(MAX_AMP);

  // Determine cross correlation
  double cross_corr = CrossCorrelation(tgt_audio);

  return 0.333 * cross_corr + 0.333 * ave_energy_diff + 0.333 * max_diff;
}

double InstumentTrainerC::CrossCorrelation(
    const std::vector<int16_t>& tgt_audio) {
  // Get next power of 2
  auto signal_length = static_cast<uint32_t>(pow(
      2, ceil(log(src_audio_.size() * 2) / log(2))));

  // Pad signals with { 0.0 0.0j }
  std::vector<std::complex<double>> cmplx_src(signal_length, { 0.0, 0.0 });
  std::vector<std::complex<double>> cmplx_tgt(signal_length, { 0.0, 0.0 });
  std::vector<std::complex<double>> cmplx_corr(signal_length, { 0.0, 0.0 });
  for (size_t s = 0; s < src_audio_.size(); s++) {
    cmplx_src[s] = src_audio_[s];
    cmplx_tgt[s] = tgt_audio[s];
  }

  Fft::convolve(cmplx_src, cmplx_tgt, cmplx_corr);

  return src_audio_.size() / (std::abs(cmplx_corr[src_audio_.size()]));
}

/*
 * Calculate the average mean error for the generated
 * signals of each trainee instrument in the class.
 */
void GeneticInstumentTrainerC::DetermineFitness() {
  bool have_new_best = false;
  double ave_error = 0.0;
  double min_error = std::numeric_limits<double>::max();
  std::vector<int16_t> best_instrument_sample(src_audio_.size());

  // TODO(Brandon): Replace with midi.
  sustain = std::vector<bool>(src_audio_.size(), true);

  // Determine error score for each instrument for class.
  std::vector<int16_t> temp_sample(src_audio_.size());
  for (size_t i = 0; i < trainees_.size(); i++) {
    // Only re-calculate score if we have not already done so.
    if (!trainees_[i]->score_is_cached_) {
      temp_sample = trainees_[i]->GenerateIntSignal(velocity, base_frequency,
                                                          src_audio_.size(),
                                                          sustain);
      trainees_[i]->error_score_ = GetError(temp_sample);
      trainees_[i]->score_is_cached_ = true;
    }

    // Add score to average if it is not infinite
    if (trainees_[i]->error_score_ < std::numeric_limits<double>::max()) {
      ave_error += trainees_[i]->error_score_;

      if (trainees_[i]->error_score_ < min_error) {
        min_error = trainees_[i]->error_score_;

        // Determine best score and if it is a new best score
        if (i > 0) {
          have_new_best = true;

          for (size_t j= 0; j < temp_sample.size(); j++){
            best_instrument_sample[j] = temp_sample[j];
          }
        }
      }
    }
  }
  ave_error = ave_error / trainees_.size();

  // Log progression and write out best sample if it is not new.
  if (!progress_location_.empty()) {
    std::cout << gen_count_ << ", " << min_error << ", " << ave_error
              << std::endl;

    // TODO(Brandon) write JSON to file.
    if (have_new_best) {
      MonoWaveWriterC wave_writer(best_instrument_sample);
      wave_writer.Write(
          progress_location_ + "/Gen_" + std::to_string(gen_count_) + ".wav");
    }
  }
}

/*
 * Determine which trainees survived this generation,
 * Replace killed off instruments with mutated survivors.
 */
void GeneticInstumentTrainerC::GeneticAlgorithm() {
  // Sort trainees. Then replace the bottom 75% trainees.
  std::sort(trainees_.begin(), trainees_.end(), instrument::cmp_by_name);
  size_t keep_amount = trainees_.size() / 3;
  for (size_t i = 0; i < trainees_.size(); i++) {
    size_t keep_index = i % (keep_amount);
    if (i >= keep_amount) {
      trainees_[i].reset(trainees_[keep_index]->TuneInstrument(1).release());
    }
  }

}

void GeneticInstumentTrainerC::Start(uint16_t a_num_of_generations) {
  num_generations_ = a_num_of_generations;

  // Pre-calculate energy of src wave
  src_energy_ = 0.0;
  for (size_t s = 0; s < src_audio_.size(); s++) {
    src_energy_ += std::abs(src_audio_[s]);
  }
  src_energy_ = src_energy_ / src_audio_.size();
  if( src_energy_ == 0 ){
    exit(EXIT_BAD_SOURCE_SIGNAL);
  }
  std::cout << "src average energy: " << src_energy_ << std::endl;

  // TODO(Brandon): Log to separate file.
  std::cout << "Generation, " << "Top instrument error, " << "Average error "
            << std::endl;
  for (gen_count_ = 0; gen_count_ < num_generations_; gen_count_++) {
    DetermineFitness();
    GeneticAlgorithm();

    // Add additional oscillator.
    if (gen_count_ != 0 && gens_per_addition_ > 0
        && gen_count_ % gens_per_addition_ == 0) {
      for (size_t i = 0; i < trainees_.size(); i++) {
        trainees_[i]->score_is_cached_ = false;
        trainees_[i]->error_score_ = std::numeric_limits<double>::max();
        trainees_[i]->AddUntunedString();
      }
    }
  }
}
}  // namespace trainer
