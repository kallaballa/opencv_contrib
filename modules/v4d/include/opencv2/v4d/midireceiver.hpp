#ifndef SRC_SLIDE_MIDI_HPP_
#define SRC_SLIDE_MIDI_HPP_
#include <opencv2/core/cvdef.h>

#include <rtmidi/RtMidi.h>
#include <iostream>
#include <mutex>
#include <chrono>
#include <vector>
#include <cstdint>

CV_EXPORTS struct MidiEvent {
	bool on_ = false;
	bool cc_ = false;
	bool clock_ = false;
	uint16_t note_ = 0;
	uint16_t velocity_ = 0;
	uint16_t channel_ = 0;
	uint64_t timestamp_ = 0;
	uint16_t controller_ = 0;
	uint16_t value_ = 0;
};

CV_EXPORTS std::ostream& operator<<(std::ostream &out, const MidiEvent& ev);
CV_EXPORTS class MidiReceiver {
private:
	RtMidiIn *midiin_ = new RtMidiIn();
	int32_t inport_;
public:
	static std::vector<MidiEvent>* queue_;
	static std::mutex* evMtx_;
	CV_EXPORTS MidiReceiver(int32_t inport, bool autostart = true);
	CV_EXPORTS virtual ~MidiReceiver();
	CV_EXPORTS void start();
	CV_EXPORTS void stop();
	CV_EXPORTS void clear();

	CV_EXPORTS std::vector<MidiEvent> receive();
};

#endif /* SRC_SLIDE_MIDI_HPP_ */
