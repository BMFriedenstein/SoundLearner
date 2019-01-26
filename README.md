# SoundLearner #
*A machine learning approach to modelling musical instruments or sounds
- - - -

##  Summary and Goals ##
This projects contains two applications:-

1. trainer
2. player

The goal of the project is to use machine learning to develop instrument model capable of accurately synthesizing the sound of actual instrument recordings. Then Using a raspberry pi and a midi instrument, generate the synthesized audio in real time to a audio device.

## Milestones ##

#### Trainer App ####
* [x] Develop simplified *oscillator* model
* [x] Develop oscillator based *Instrument model*
* [x] Add basic wave file read and write
* [ ] Add basic mid/midi file read
* [x] Add Json based *Instrument model* save
* [x] Build Reinforcement learning based training framework
* [x] Train instrument model with piano sound sample
* [ ] Build Deep learning based training framework
* [x] Build Deep learning Dataset
* [ ] Train generlised deep leaning instrument generator
* [ ] Test instrument generator with real instrument sounds
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
