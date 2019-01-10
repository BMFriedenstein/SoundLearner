/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 *
 * trainer.cpp
 *
 *  Created on: 04 Jan 2019
 *      Author: brandon
 */

#include "trainer.h"

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
// #include <thread>

#include "wave/wave.h"

namespace instrument{
    inline bool cmp_by_name(const std::unique_ptr<InstrumentModelC>& a, const std::unique_ptr<InstrumentModelC>& b)
    {
        return *a < *b;
    }
}

namespace trainer{
InstumentTrainerC::InstumentTrainerC(
        uint16_t num_starting_occilators,
        uint16_t class_size,
        std::vector<int16_t>& source_audio,
        std::string& progress_location){
    progress_location_ = progress_location;
    source_audio_ = source_audio;
    trainees_.reserve(class_size);
    for(uint16_t i=0; i< class_size; i++){
        std::string name_of_instrument = "instrument_" + std::to_string(i);
        std::unique_ptr<instrument::InstrumentModelC> new_instrument(
            new instrument::InstrumentModelC(num_starting_occilators, name_of_instrument)
        );
        trainees_.push_back(std::move(new_instrument));
    }
}

GeneticInstumentTrainerC::GeneticInstumentTrainerC(
        uint16_t num_starting_occilators,
        uint16_t class_size,
        std::vector<int16_t>& source_audio,
        std::string& progress_location,
        uint32_t gens_per_addition)
: trainer::InstumentTrainerC(num_starting_occilators, class_size, source_audio, progress_location)
, gens_per_addition_(gens_per_addition) {
    std::vector<bool> sustain = std::vector<bool>(source_audio.size(), true);
}


/*
 * Calculate the average mean error for the generated signals of each trainee instrument
 * in class
 */
double InstumentTrainerC::GetError(const std::vector<int16_t>& instrument_audio){

    // Check the source audio is the right size
    if( instrument_audio.size() != source_audio_.size() ){
        std::cout << "WARN !!! BAD audio size" << std::endl;
        return std::numeric_limits<double>::max();
    }

    // Get energy in sample
    long long sample_energy_int = 0;
    for (size_t s = 0; s<instrument_audio.size(); s++) {
        sample_energy_int += instrument_audio[s];
    }
    double sample_energy = static_cast<double>(sample_energy_int);

    // Calculate corrected mean absolute error
    double energy_correction = source_energy_/sample_energy;
    double sum_abs_error = 0;
    for (size_t s=0; s<source_audio_.size(); s++){
        sum_abs_error += std::abs(
            energy_correction*static_cast<double>(source_audio_[s]) - static_cast<double>(instrument_audio[s])
        );
    }

    // Combine mean error and energy differential
    double final_error = (0.5*sum_abs_error/static_cast<double>(source_audio_.size()) ) +
                         (0.5*std::abs(source_energy_- sample_energy));
    return final_error;
}

/*
 * Calculate the average mean error for the generated signals of each trainee instrument
 * in the class.
 */
void GeneticInstumentTrainerC::DetermineFitness(){
    double ave_error = 0.0;
    double min_error = std::numeric_limits<double>::max();
    std::vector<int16_t> best_instrument_sample(source_audio_.size());
    sustain = std::vector<bool>(source_audio_.size(),true); // TODO

    // Determine error score for each instrument for class
    std::vector<int16_t> temp_sample;
    for (size_t i=0; i<trainees_.size(); i++) {
        trainees_[i]->error_score_ = std::numeric_limits<double>::max();

        // TODO(BRANDON): Add threading here to improve performance
        temp_sample = trainees_[i]->GenerateIntSignal(
            velocity,
            base_frequency,
            source_audio_.size(),
            sustain
        );

        trainees_[i]->error_score_= GetError( temp_sample );
        ave_error += trainees_[i]->error_score_/trainees_.size();

        if(trainees_[i]->error_score_ < min_error){
            min_error = trainees_[i]->error_score_;
            best_instrument_sample = temp_sample;
        }
    }

    // Log progression and write out best sample
    if(!progress_location_.empty()){
        std::cout<< gen_count_
                       << ", " << min_error
                       << ", " << ave_error
                       << std::endl;
        // TODO write JSON to file
        MonoWaveWriterC wave_writer(best_instrument_sample);
        wave_writer.Write(progress_location_ + "/Gen_" + std::to_string(gen_count_) + ".wav");
    }
}

/*
 * Determine which trainees survived this generation,
 * Replace killed off instruments with mutated survivors
 */
void GeneticInstumentTrainerC::GeneticAlgorithm(){

    // Sort trainees
    std::sort(trainees_.begin(), trainees_.end(), instrument::cmp_by_name);
    size_t keep_amount = trainees_.size()/5L;
    for (size_t i=0; i<trainees_.size(); i++) {
        size_t keep_index = i%(keep_amount);
        if(i > keep_amount){
            trainees_[i].reset(
                trainees_[keep_index]->TuneInstrument(100).release() // 100 out of max 255
            );
        }
    }
}

void GeneticInstumentTrainerC::Start(uint16_t a_num_of_generations) {
    num_generations_ = a_num_of_generations;
    long long source_energy_int=0;
    for (size_t s = 0; s < source_audio_.size(); s++) {
        source_energy_int += source_audio_[s];
    }
    source_energy_ = static_cast<double>(source_energy_int);

    // TODO(BRANDON): Log to sepperate file
    std::cout  << "Generation, "
               << "Top instrument error, "
               << "Average error "
               <<  std::endl;

    for (gen_count_=0; gen_count_<num_generations_; gen_count_++) {
        DetermineFitness();
        GeneticAlgorithm();
        // Add additional oscillator
        if( gen_count_!=0 && gen_count_%gens_per_addition_==0 ){
            for(size_t i=0; i<trainees_.size(); i++){
                trainees_[i]->AddUntunedString();
            }
        }

    }
}
} // namespace trainer
