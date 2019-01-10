/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * trainer.h
 *
 *  Created on: 04 Jan 2019
 *      Author: brandon
 */

#ifndef STRING_SOUNDS_TRAINER_TRAINER_H_
#define STRING_SOUNDS_TRAINER_TRAINER_H_

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <climits>

#include "instrument/instrument_model.h"

namespace trainer{
// Basic training framework
class InstumentTrainerC{
protected:
   std::vector<std::unique_ptr<instrument::InstrumentModelC>> trainees_;
   std::string progress_location_;
   std::vector<int16_t> source_audio_;
   double source_energy_ = 0;

public:
   virtual double GetError(const std::vector<int16_t>& instrument_audio);
   InstumentTrainerC(
       uint16_t num_starting_occilators,
       uint16_t class_size,
       std::vector<int16_t>& source_audio,
       std::string& progress_location
   );
   virtual ~InstumentTrainerC(){  }
};

// A training framework based on a genetic algorithm
class GeneticInstumentTrainerC : public InstumentTrainerC{
private:
   uint32_t    gens_per_addition_;
   uint16_t    num_generations_=0;
   uint16_t    gen_count_  = 0;

   // TODO replace with MIDI input
   double base_frequency = 98.0;
   std::vector<bool> sustain;
   double velocity = 1.0;

   void GeneticAlgorithm();
   void DetermineFitness();

public:
   GeneticInstumentTrainerC(
       uint16_t num_starting_occilators,
       uint16_t class_size,
       std::vector<int16_t>& source_audio,
       std::string& progress_location,
       uint32_t gens_per_addition
   );
   ~GeneticInstumentTrainerC(){  }
   void Start(const uint16_t a_num_of_generations);
};

// TODO other training frameworks
}

#endif // STRING_SOUNDS_TRAINER_TRAINER_H_
