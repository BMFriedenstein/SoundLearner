/*
 * main.cpp
 *
 *  Created on: 02 Jan 2019
 *      Author: brandon
 */

#include <stdio.h>
#include <iostream>

#include "include/common.h"
#include "instrument/instrument_model.h"
#include "instrument/string_oscillator.h"
#include "trainer.h"

static void  AppUsage(){
    std::cerr << "Usage: \n"
              << "-n --size <100 instruments> \n"
              << "-t --time <train for 10000s> \n"
              << "-a --target <train.wav> \n"
              << "-m --midi <train.mid>\n"
              << "-s --instrument-size <100>\n"
              << "-M --max-instrument-size <1000>\n"
              << "-r --instrument-growth-rate <0> (generations per increase)\n"
              << "-m --midi <train.mid>\n"
              << "-p --progression <./training/> (save progresion)"
              << std::endl;
}

int main(int argc, char** argv){

    // Application defaults
    uint16_t class_size = 100;
    uint16_t start_instrument_size = 10;
    uint16_t max_instrument_size = 1000;
    uint16_t instrument_growth = 0;
    uint32_t training_time = 10000;
    std::string audio_file = "train.wav";
    std::string midi_file = "train.mid";
    bool save_progression = true;
    std::string progression_output = "./progression/";

    // Parse arguments
    for ( int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            AppUsage();
            return EXIT_NORMAL;
        }
        if (((arg == "-n") || (arg == "--size") ||
            (arg == "-t") || (arg == "--time") ||
            (arg == "-a") || (arg == "--target") ||
            (arg == "-m") || (arg == "--midi") ||
            (arg == "-r") || (arg == "--instrument-growth-rate") ||
            (arg == "-s") || (arg == "--instrument-size") ||
            (arg == "-M") || (arg == "--max-instrument-size") ||
            (arg == "-p") || (arg == "--progression")) &&
            (i + 1 < argc)) {
            std::string arg2 = argv[++i];
            if( (arg == "-n") || (arg == "--size")){
                class_size = (uint16_t)std::stol(arg2);
            }
            else if ((arg == "-t") || (arg == "--time")){
                training_time = (uint32_t)std::stoul(arg2);
            }
            else if ((arg == "-a") || (arg == "--target")){
                audio_file = arg2;
            }
            else if ((arg == "-m") || (arg == "--midi")){
                midi_file = arg2;
            }
            else if ((arg == "-r") || (arg == "--instrument-growth-rate")){
                instrument_growth = (uint16_t)std::stol(arg2);
            }
            else if ((arg == "-s") || (arg == "--instrument-size")){
                start_instrument_size = (uint16_t)std::stol(arg2);
            }
            else if ((arg == "-M") || (arg == "--max-instrument-size")){
                max_instrument_size = (uint16_t)std::stol(arg2);
            }
            else if ((arg == "-p") || (arg == "--progression")){
                save_progression = true;
                progression_output = arg2;
            }
        }
        else {
            std::cerr << "--destination option requires one argument."  << std::endl;
            return EXIT_BAD_ARGS;
        }
    }


    // Sanity check
    if( start_instrument_size > max_instrument_size ){ start_instrument_size = max_instrument_size; }

    // Some debug
	std::cout << "Starting trainer... " << std::endl;
	std::cout << "\tTraining time: " << training_time << " seconds" << std::endl;
	std::cout << "\tSource audio: " << audio_file << std::endl;
	std::cout << "\tSource MIDI: " << midi_file << std::endl;
	std::cout << "\tClass size: " << class_size << std::endl;
	std::cout << "\tStarting instrument size: " << start_instrument_size << std::endl;
	std::cout << "\tMaximum instrument size: " << max_instrument_size << std::endl;
	std::cout << "\tInstrument growth rate: " << instrument_growth << " generations per increase" << std::endl;
	if(save_progression){
	    std::cout << "\tProgression saved to: " << progression_output << std::endl;
	}

	// Todo Read source wav file into memory
	// Todo Read midi file into memory
	// Todo Create models
	// Todo create trainer class
	// Todo start training
	return EXIT_NORMAL;
}

