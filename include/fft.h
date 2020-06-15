/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * fft.h
 *  Created on: 03 Jan 2019
 *      Author: Brandon
 */

#pragma once
#ifndef INCLUDE_FFT_H_
#define INCLUDE_FFT_H_

#include <fftw3.h>
#include <cmath>
#include <cstddef>
#include <cstring>

#include <algorithm>
#include <array>
#include <complex>
#include <vector>

#include "include/common.h"

namespace fft {
static constexpr uint8_t real = 0;
static constexpr uint8_t imag = 1;

enum WindowTypes {
  NONE,
  HANN,
  FLAT_TOP,
  UNIFORM,
  FORCE,
  HAMMING,
  KAISER_BESSEL,
  EXPONENTIAL
};

template <typename T, std::size_t A>
static inline void WindowFunction(std::array<T, A>* in_out_data, WindowTypes window_function) {
  static_assert(std::is_arithmetic<T>::value, "Not an arithmetic type");
  switch (window_function) {
    case HANN: {
      constexpr double scale_val = M_PI / A;
      std::size_t i{0};
      std::transform(in_out_data->begin(), in_out_data->end(), in_out_data->begin(), [&i, &scale_val](auto in_val) {
        (void)in_val;
        return T(std::pow(std::sin(i++ / scale_val), 2));
      });
    } break;

    // TODO(brandon): Handle other window functions
    case FLAT_TOP:
    case UNIFORM:
    case FORCE:
    case HAMMING:
    case KAISER_BESSEL:
    case EXPONENTIAL:
    case NONE:
    default:
      break;
  }
}

template <typename T, std::size_t A>
static void OneSidedFFT(std::array<T, A>* in_out_data, const double& freq, WindowTypes window_function = HANN) {
  static_assert(std::is_arithmetic<T>::value, "Not an arithmetic type");
  constexpr std::size_t in_fft_size = 2 * A;
  WindowFunction<T, A>(in_out_data, window_function);
  std::array<fftw_complex, in_fft_size> in;
  std::array<fftw_complex, in_fft_size> out;
  for (std::size_t i{0}; i < in_fft_size; ++i) {
    in[i][real] = i < A ? (*in_out_data)[i] : 0;
    in[i][imag] = 0;
  }

  fftw_plan p = fftw_plan_dft_1d(static_cast<int>(in_fft_size), in.data(), out.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);

  for (std::size_t i = 0; i < A; ++i) {
    const double magnitude{std::abs(std::complex<double>(out[i][real], out[i][imag]))};
    const double power_density{std::pow(magnitude, 2) / (2 * A * freq)};
    const double log_val{10 * std::log10(power_density)};
    (*in_out_data)[i] = static_cast<T>(log_val);
  }
  fftw_destroy_plan(p);
  fftw_cleanup();
}

namespace spectrogram {

// static std::vector<double>
// LogTransform(const uint32_t tgt_resolution,
//              std::vector<std::complex<double>> &vec) {
//   (void)tgt_resolution;
//   (void)vec;
//   return std::vector<double>(); // TODO(brandon): transform spectogram to logarithmic
// }

template <typename Ti, typename To, std::size_t R>
static std::array<std::array<To, R>, R> CreateSpectrogram(const std::vector<Ti>& source_signal,
                                                          const Ti& min_val,
                                                          const Ti& max_val) {
  static_assert(std::is_arithmetic<Ti>::value, "Not an arithmetic type");
  static_assert(std::is_arithmetic<To>::value, "Not an arithmetic type");
  const double slope_m = -1 / (min_val - max_val);
  auto return_spectogram = std::array<std::array<To, R>, R>();
  const std::size_t increment_size = source_signal.size() / R;

  std::size_t source_idx = 0;
  for (auto window = return_spectogram.begin(); window != return_spectogram.end(); ++window) {
    std::array<Ti, R> frame;
    std::memset(frame.data(), 0, frame.size());
    std::memcpy(frame.data(), &source_signal[source_idx], std::min(frame.size(), source_signal.size() - source_idx));
    OneSidedFFT<Ti, R>(&frame, SAMPLE_RATE);

    // Normalize to values between 0 and 1
    std::transform(window->begin(), window->end(), window->begin(), [&](const auto& val) {
      return static_cast<To>(std::clamp<double>(val * slope_m, min_val, max_val));
    });
    source_idx += increment_size;
  }
  return return_spectogram;
}
}  // namespace spectrogram
}  // namespace fft
#endif /* INCLUDE_FFT_H_ */
