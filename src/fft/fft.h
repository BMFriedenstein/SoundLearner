#ifndef SRC_FFT_FFT_H_
#define SRC_FFT_FFT_H_

#include "shared/common.h"
#include <bits/stdint-uintn.h>
#include <array>
#include <cstddef>
#include <vector>

namespace Fft {
const uint8_t real = 0;
const uint8_t imag = 1;
enum WindowTypes {
   NONE,
   HANN,
   FLAT_TOP,
   UNIFORM,
   FORCE,
   HAMMING,
   KAISER_BESSEL,
   EXPONENTIAL
   // TODO others
};
void WindowFunction(std::vector<double>& in_out_data, WindowTypes window_function);
void OneSidedFFT(std::vector<double>& in_out_data,
                 uint32_t fft_size,
                 const double freq,
                 WindowTypes window_function = HANN);


namespace spectrogram {
std::vector<double> LogTransform(const uint32_t tgt_resolution, std::vector<double>& vec);
template<typename T>
std::vector<std::vector<double>> CreateSpectrogram(const std::vector<T>& source_signal,
                                                   const uint32_t resolution,
                                                   const double min_val,
                                                   const double max_val){
   double slope_m = -1 / (min_val - max_val);
   auto return_spectogram = std::vector<std::vector<double>>(resolution);
   auto begin_iter = source_signal.begin();
   auto end_iter = source_signal.begin();
   uint32_t increment_size = source_signal.size()/resolution;

   for (auto window = return_spectogram.begin(); window != return_spectogram.end(); ++window) {
      end_iter += increment_size;
      *window = std::vector<double>(begin_iter, end_iter);
      Fft::OneSidedFFT(*window, resolution, SAMPLE_RATE);

      // Normalize to values between 0 and 1
      for (auto val_iter = window->begin(); val_iter != window->end(); ++val_iter) {
         *val_iter *= slope_m;
         if (*val_iter > max_val) {
            *val_iter = 1;
         }
         else if (*val_iter < min_val) {
            *val_iter = 0;
         }
      }
      begin_iter = end_iter;
   }
   return return_spectogram;
}
}
}
#endif /* SRC_FFT_FFT_H_ */
