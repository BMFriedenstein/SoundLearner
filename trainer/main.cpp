/*
 * main.cpp
 *
 *  Created on: 02 Jan 2019
 *      Author: brandon
 */

#include <stdio.h>
#include <iostream>
#include <common/common.h>
#include <common/model.h>
#include <common/sound_string.h>

#include "trainer.h"
std::string AppUsage(){
    return "trainer -n <class_size (100)> -t <training time (10000s)> -s <audio file (train.wav)> -m <corresponding midi file (train.mid)> -p <Save Progression true>";
}

int main(int argc, char** argv){

    // Application defaults
    uint16_t class_size = 100;
    uint32_t training_time = 10000;
    std::string audio_file = "train.wav";
    std::string midi_file = "train.mid";
    bool save_progression = true;

    // TODO parse arguments
    /*int i;
    for (i = 0; i < argc; i++){

    }
    */
	std::cout << "Starting trainer... " << training_time << " seconds remaining" << std::endl;

	(void)class_size;
	(void)training_time;
	(void)audio_file;
	(void)midi_file;
	(void)save_progression;
	return EXIT_NORMAL;
}

