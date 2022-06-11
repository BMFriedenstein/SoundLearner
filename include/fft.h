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
static inline constexpr uint8_t real = 0;
static inline constexpr uint8_t imag = 1;
static double fft_spectogram_min = -70.0;
static double fft_spectogram_max = 50.0;

enum class WindowTypes {
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
    case WindowTypes::HANN: {
      constexpr double scale_val = M_PI / A;
      std::size_t i{0};
      std::transform(in_out_data->begin(), in_out_data->end(), in_out_data->begin(), [&i, &scale_val](auto in_val) {
        return T(in_val*std::pow(std::sin(i++ / scale_val), 2));
      });
    } break;

    // TODO(brandon): Handle other window functions
    case WindowTypes::FLAT_TOP:
    case WindowTypes::UNIFORM:
    case WindowTypes::FORCE:
    case WindowTypes::HAMMING:
    case WindowTypes::KAISER_BESSEL:
    case WindowTypes::EXPONENTIAL:
    case WindowTypes::NONE:
    default:
      break;
  }
}

template <typename T, std::size_t A>
static inline void OneSidedFFT(std::array<T, A>* in_out_data, double freq, WindowTypes window_function = WindowTypes::NONE) {
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
    const double log_val{-10 * std::log10(power_density)};
    (*in_out_data)[i] = static_cast<T>(log_val);
  }
  fftw_destroy_plan(p);
  fftw_cleanup();
}

namespace spectrogram {

template <typename Ti, typename To, std::size_t X, std::size_t Y=X>
static inline std::array<std::array<To, Y>, X> CreateSpectrogram(const std::vector<Ti>& source_signal) {
  static_assert(std::is_arithmetic<Ti>::value, "Not an arithmetic type");
  static_assert(std::is_arithmetic<To>::value, "Not an arithmetic type");
  const To max = std::is_floating_point<To>::value ? To(1.0) : std::numeric_limits<To>::max();
  const double slope_m = static_cast<double>(max / (fft_spectogram_min - fft_spectogram_max));

  std::array<Ti, Y> frame;
  std::array<std::array<To, Y>, X> return_spectogram = {{0}};
  const std::size_t increment_size = source_signal.size() / X;
  std::size_t source_idx = 0;
  for (auto window = return_spectogram.begin(); window != return_spectogram.end(); ++window) {
    std::memset(frame.data(), 0, sizeof(Ti) * frame.size());
    std::memcpy(frame.data(), &source_signal[source_idx], sizeof(Ti) * std::min(frame.size(), source_signal.size() - source_idx));
    OneSidedFFT<Ti, Y>(&frame, SAMPLE_RATE);

    // Normalize to values between 0 and 1
    std::transform(frame.begin(), frame.end(), window->begin(), [&](const auto& val) {
      return static_cast<To>(std::clamp<double>(val * slope_m, 0.0, max));
    });
    source_idx += increment_size;
  }
  return return_spectogram;
}

template <typename Ti, typename To, std::size_t R, std::size_t M=R>
static inline std::array<std::array<To, M>, M> CreateMelSpectrogram(const std::vector<Ti>& source_signal) {
  static_assert(std::is_arithmetic<Ti>::value, "Not an arithmetic type");
  static_assert(std::is_arithmetic<To>::value, "Not an arithmetic type");
  constexpr std::size_t Q = M * 25;


  const double max = std::is_floating_point<To>::value ? 1.0 : static_cast<double>(std::numeric_limits<To>::max());
  const double slope_m = max / (fft_spectogram_min - fft_spectogram_max);

  // Mel Transform
  const double log_max = std::log2(static_cast<double>(SAMPLE_RATE));
  const double log_increments = log_max / M;

  std::array<std::array<To, M>, M> mel_spectogram{{0}};
  std::size_t source_idx = 0;
  const std::size_t increment_size = source_signal.size() / M;

  std::array<Ti, Q> frame = {0};
  for (std::size_t x = 0; x < M; ++x) {
    std::memset(frame.data(), 0, sizeof(Ti) * frame.size());
    std::memcpy(frame.data(), &source_signal[source_idx], sizeof(Ti) * std::min(R*4, source_signal.size() - source_idx));
    OneSidedFFT<Ti, Q>(&frame, SAMPLE_RATE);

    std::array<std::vector<Ti>, M> log_bins{};
    for (std::size_t y = 0; y < frame.size(); ++y) {
      std::size_t mapped_y = y > 0 ? static_cast<std::size_t>(std::round(std::log2(y) / log_increments)) : 0;
      mapped_y = std::clamp(mapped_y, std::size_t(0), std::size_t(M - 1));
      log_bins[mapped_y].push_back(frame[y]);
    }

    for (std::size_t y = 0; y < M; ++y) {
      if (log_bins[y].empty()) {
        mel_spectogram[x][y] = 0;
      } else {
        const double sum_val = std::accumulate(log_bins[y].begin(), log_bins[y].end(), 0);
        mel_spectogram[x][y] = static_cast<To>(std::clamp(slope_m * sum_val / log_bins[y].size(), 0.0, max));
      }
    }
    source_idx += increment_size;
  }
  return mel_spectogram;
}

}  // namespace spectrogram
}  // namespace fft
#endif /* INCLUDE_FFT_H_ */
