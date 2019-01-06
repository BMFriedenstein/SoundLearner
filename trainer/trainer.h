/*
 * trainer.h
 *
 *  Created on: 04 Jan 2019
 *      Author: brandon
 */

#ifndef TRAINER_TRAINER_H_
#define TRAINER_TRAINER_H_

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <climits>
// #include <thread>
#include "instrument/instrument_model.h"

class InstumentTrainerC{
private:
    struct TraineeInstrument{
        std::unique_ptr<InstrumentModelC> instrument;
        double score = 2*MAX_AMP;
    };
    std::vector<int16_t> source_audio;
    std::map<std::string,TraineeInstrument> trainee_instruments;
    uint16_t class_size;
    uint32_t gens_per_addition;
    uint16_t num_of_generations;
    std::string progress_location;

    // TODO replace with MIDI input
    double base_frequency = 98.0;
    std::vector<bool> sustain;
    double velocity = 1;

    void TrainGeneration( uint16_t gen_count );
    double GetError( std::vector<int16_t>& instrument_audio );

public:
    InstumentTrainerC( uint16_t a_class_size,
                       uint16_t a_starting_occilators,
                       std::vector<int16_t>& a_source_audio,
                       uint16_t a_num_of_gens=10,
                       uint32_t a_gens_per_addtion = 0xffffffff,
                       std::string a_instrument_name="instrument",
                       std::string progress_location="progression/");
    void Start( uint16_t a_num_of_generations );
};

#endif /* TRAINER_TRAINER_H_ */
