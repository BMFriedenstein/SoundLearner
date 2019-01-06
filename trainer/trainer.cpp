/*
 * trainer.cpp
 *
 *  Created on: 04 Jan 2019
 *      Author: brandon
 */

#include <math.h>
#include <stdio.h>
#include <iostream>
#include "wave/wave.h"
#include "trainer.h"
using namespace std;

InstumentTrainerC::InstumentTrainerC( uint16_t a_class_size,
                                      uint16_t a_starting_occilators,
                                      vector<int16_t>& a_source_audio,
                                      uint16_t a_num_of_gens,
                                      uint32_t a_gens_per_addtion,
                                      string a_instrument_name,
                                      string a_progress_location ) {
    progress_location = a_progress_location;
    source_audio = a_source_audio;
    class_size = a_class_size;
    gens_per_addition = a_gens_per_addtion;
    num_of_generations = a_num_of_gens;
    vector<bool> sustain = vector<bool>( a_source_audio.size(), true );
    for( uint16_t i=0; i< class_size; i++ ){
        string name_of_instrument = a_instrument_name + "_" + to_string(i);
        TraineeInstrument new_instrument_trainee;
        new_instrument_trainee.instrument = unique_ptr<InstrumentModelC>(
            new InstrumentModelC( a_starting_occilators, name_of_instrument )
        );
        trainee_instruments.insert( make_pair(name_of_instrument, move(new_instrument_trainee)) );
    }
}

/*
 *
 */
double InstumentTrainerC::GetError( std::vector<int16_t>& instrument_audio ){
    if( instrument_audio.size() != source_audio.size() ){
        cout << "WARN !!! BAD audio size" << endl;
        return 2*MAX_AMP;
    }

    uint64_t sum_abs_error = 0;
    for ( size_t s=0; s< source_audio.size(); s++ ){
        sum_abs_error += abs(source_audio[s] - instrument_audio[s]);
    }

    return (double)sum_abs_error/(double)source_audio.size();
}

/*
 *
 */
void InstumentTrainerC::TrainGeneration( uint16_t gen_count ){
    double ave_error = 0;
    double min_error = 3*MAX_AMP;
    std::string best_instrument;

    // Determine errors
    for ( auto iter = trainee_instruments.begin(); iter != trainee_instruments.end(); iter++ ) {
        vector<int16_t> instr_signal = iter->second.instrument->GenerateIntSignal(
                velocity, base_frequency, source_audio.size(), sustain );
        double error = GetError( instr_signal );
        iter->second.score = error;
        ave_error += error;
        if( error < min_error ){
            min_error = error;
            best_instrument = iter->first;
        }
    }

    // Log progression
    if( !progress_location.empty() ){
        cout << "Generation " << gen_count
                       << ": Top instrument error: " << min_error
                       << ", Average error " << ave_error
                       // << "\n" << trainee_instruments[best_instrument].instrument->ToJson()
                       << " ... " << endl;
        // TODO write JSON to file
        vector<int16_t> instr_signal = trainee_instruments[best_instrument].instrument->GenerateIntSignal(
                velocity, base_frequency, source_audio.size(), sustain );
        MonoWaveWriterC wave_writer( instr_signal );
        wave_writer.Write(progress_location + "/Gen_" + to_string(gen_count) + ".wav");
    }

    // Do mutations
    for ( auto iter = trainee_instruments.begin();iter != trainee_instruments.end(); iter++ ) {
        if( iter->second.score <=  ave_error ){
            unique_ptr<InstrumentModelC> mut_instrument = trainee_instruments[best_instrument].instrument->TuneInstrument( 20 );
            iter->second.instrument.reset(mut_instrument.release());
        }
    }
}

/*
 *
 */
void InstumentTrainerC::Start( uint16_t a_num_of_generations ) {

    // TODO threading
    for (uint16_t gen = 0; gen < num_of_generations; gen++) {
        TrainGeneration(gen);

        // Add additional oscillator
        if( gen!=0 && gen%gens_per_addition==0 ){
            for( auto iter = trainee_instruments.begin();iter != trainee_instruments.end(); iter++ ){
                iter->second.instrument->AddUntunedString();
            }
        }
    }
}
