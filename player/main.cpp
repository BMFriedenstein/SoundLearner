/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * main.cpp
 *  Created on: 02 Jan 2019
 *      Author: Brandon
 */

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "include/common.h"
#include "include/filereader.h"
#include "include/filewriter.h"
#include "instrument/instrument_model.h"
#include "instrument/string_oscillator.h"

static void AppUsage() {
  std::cerr << "Usage: \n"
            << "-h --help\n"
            << "-f --filename <'instrument'> \n"
            << "-n --note<440>\n"
            << "-v --velocity<100>\n"
            << "-l --length<5s>\n"
            << std::endl;
}

int main(int argc, char** argv) {
  double velocity = 1.0;
  double note_played = 440.0;
  std::string filename = "";
  uint32_t num_samples = 5 * 44100;
  // Parse arguments.
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      AppUsage();
      return EXIT_NORMAL;
    }
    if (((arg == "-f") || (arg == "--filename") || (arg == "-n") || (arg == "--note") || (arg == "-v") ||
         (arg == "--velocity") || (arg == "-l") || (arg == "--length")) &&
        (i + 1 < argc)) {
      std::string arg2 = argv[++i];
      std::cout << arg << " " << arg2 << std::endl;
      if ((arg == "-f") || (arg == "--filename")) {
        filename = arg2;
      } else if ((arg == "-n") || (arg == "--note")) {
        note_played = std::stof(arg2);
      } else if ((arg == "-v") || (arg == "--velocity")) {
        velocity = ((uint8_t)std::stoi(arg2)) / 100.0;
      } else if ((arg == "-l") || (arg == "--length")) {
        num_samples = ((uint32_t)std::stoul(arg2)) * 44100;
      } else {
        std::cerr << "--destination option requires one argument." << std::endl;
        return EXIT_BAD_ARGS;
      }
    }
  }

  // Now read the model
  std::vector<std::string> instrument_strings;
  std::ifstream file(filename);  // file just has some sentences
  if (!file) {
    std::cout << "unable to open file: " << filename << std::endl;
    return EXIT_BAD_ARGS;
  }

  std::string string_line;
  while (std::getline(file, string_line)) {
    instrument_strings.push_back(string_line);
    std::cout << string_line << std::endl;
  }

  std::cout << "\nmodel:\n" << std::endl;
  instrument::InstrumentModel instru_model(instrument_strings, filename);
  std::cout << instru_model.ToJson() << std::endl;
  bool has_distorted;
  std::vector<int16_t> sample = instru_model.GenerateIntSignal(velocity, note_played, num_samples, has_distorted);
  filewriter::wave::MonoWriter wave_writer(sample);
  wave_writer.Write(filename + ".wav");
}
