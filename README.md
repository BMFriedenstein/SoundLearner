# SoundLearner #
*A machine learning'ish approach to modelling musical instruments*
- - - -

##  Summary and Goals ##
This projects contains two applications:-

1. trainer
2. player

The goal of the project is to use machine learning to develop instrument model capable of accurately synthesizing thesound of actual instrument recordings. Then Using a raspberry pi and a midi instrument, generate the synthesized audio in real time to a audio device.

## Milestones ##

#### Trainer App ####
* [x] Develop simplified *oscillator* model
* [x] Develop oscillator based *Instrument model*
* [x] Add basic wave file read and write
* [ ] Add basic mid/midi file read
* [ ] Add Json based *Instrument model* save
* [ ] Build Reinforcement learning based training framework
* [ ] Train instrument model with piano sound sample sample
* [ ] ...* Future changes and new ideas*

#### Player ####
* [ ] Read midi input from serial USB device
* [ ] Write sound to standard audio output device
* [ ] Write sound to bluetooth audio output device
* [ ] Open json *Instrument model* files
* [ ] Playback instrument to audio output device using midi input
* [ ] Add polyphony playback
* [ ] Add stereoization
* [ ] Consider writing audio effects such as reverb, filter, chorus etc.
* [ ] ...* Future changes and new ideas*
