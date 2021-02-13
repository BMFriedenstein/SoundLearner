/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * trainer.h
 *  Created on: 04Jan 2019
 *      Author: Brandon
 */
#pragma once
#ifndef GENETIC_TRAINER_TRAINER_H_
#define GENETIC_TRAINER_TRAINER_H_

#include <climits>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "instrument/instrument_model.h"

namespace trainer {

class CandidateInstrument : public instrument::InstrumentModel {
 public:
  double error_score;
  double mae_score;
  double corr_score;
  double diff_score;
  bool score_is_cached;
  CandidateInstrument(std::size_t num_strings, const std::string& instrument_name)
      : instrument::InstrumentModel(num_strings, instrument_name),
        error_score(std::numeric_limits<double>::max()),
        mae_score(std::numeric_limits<double>::max()),
        corr_score(std::numeric_limits<double>::max()),
        diff_score(std::numeric_limits<double>::max()) {}

  CandidateInstrument(instrument::InstrumentModel&& instrument)
      : instrument::InstrumentModel(std::move(instrument)),
        error_score(std::numeric_limits<double>::max()),
        mae_score(std::numeric_limits<double>::max()),
        corr_score(std::numeric_limits<double>::max()),
        diff_score(std::numeric_limits<double>::max()) {}


  bool operator<(const CandidateInstrument& other) const { return error_score < other.error_score; }
};


// Basic training framework.
class InstumentTrainer {
 protected:
  std::string progress_location;
  std::vector<int16_t> src_audio;
  std::vector<std::unique_ptr<CandidateInstrument>> trainees;
  double src_energy;

 public:
  virtual double GetError(const std::vector<int16_t>& tgt_audio,
                          double* corr_score,
                          double* mae_score,
                          double* diff_score);
  double CrossCorrelation(const std::vector<int16_t>& tgt_audio);
  double MeanAbsoluteError(const std::vector<int16_t>& tgt_audio, double corr_factors);
  InstumentTrainer(std::size_t num_starting_occilators,
                    std::size_t class_size,
                    const std::vector<int16_t>& src_audio,
                    const std::string& progress_location);
  virtual ~InstumentTrainer() {}
};

// A training framework based on a genetic algorithm.
class GeneticInstumentTrainer : public InstumentTrainer {
 private:
  std::size_t gens_per_addition;
  std::size_t num_generations;
  std::size_t gen_count;

  // TODO(Brandon): replace with MIDI parser.
  double base_frequency = 440.0;
  double velocity = 1.0;

  void GeneticAlgorithm();
  void DetermineFitness();

 public:
  GeneticInstumentTrainer(std::size_t num_starting_occilators,
                           std::size_t class_size,
                           const std::vector<int16_t>& audio,
                           const std::string& location,
                           std::size_t gens_per_addition);
  ~GeneticInstumentTrainer() {}
  void Start(std::size_t a_num_of_generations);
};

}  // namespace trainer

#endif  // GENETIC_TRAINER_TRAINER_H_
