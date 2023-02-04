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

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <complex>
#include <limits>
#include <vector>

#include "include/common.h"

namespace fft {
static inline constexpr uint16_t max_fft_size = 0xFFFFU;
static inline constexpr uint8_t real = 0;
static inline constexpr uint8_t imag = 1;
static double fft_spectogram_min = -70.0;
static double fft_spectogram_max = 50.0;
static std::array<fftw_complex, max_fft_size> fft_in_buffer;
static std::array<fftw_complex, max_fft_size> fft_out_buffer;
static std::array<double, max_fft_size> scratch_buffer = {};

enum class WindowTypes { NONE, HANN, FLAT_TOP, UNIFORM, FORCE, HAMMING, KAISER_BESSEL, EXPONENTIAL };

template <std::size_t A> static inline void WindowFunction(std::array<double, A> *in_out_data, WindowTypes window_function) {
  switch (window_function) {
  case WindowTypes::HANN: {
    auto i{0U};
    std::transform(in_out_data->begin(), in_out_data->end(), in_out_data->begin(),
                   [&i](auto in_val) { return in_val * std::pow(std::sin(i++ / (2 * M_PI / A)), 2); });
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

template <std::size_t A>
static inline void OneSidedFFT(std::array<double, A> *in_out_data, double freq, WindowTypes window_function = WindowTypes::NONE) {
  constexpr std::size_t in_fft_size = 2 * A;
  static_assert(in_fft_size < max_fft_size);

  WindowFunction(in_out_data, window_function);

  // Prepare input/output buffers for fft.
  std::memset(fft_in_buffer.data(), 0, sizeof(fftw_complex) * 2 * A);
  std::memset(fft_out_buffer.data(), 0, sizeof(fftw_complex) * 2 * A);
  for (auto i{0U}; i < A; ++i) {
    fft_in_buffer[i][real] = (*in_out_data)[i];
  }

  // Perform fft
  fftw_plan p = fftw_plan_dft_1d(static_cast<int>(in_fft_size), fft_in_buffer.data(), fft_out_buffer.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);

  // Calculate power fft and place on return buffer.
  for (auto i{0U}; i < A; ++i) {
    const double magnitude{std::abs(std::complex<double>(fft_out_buffer[i][real], fft_out_buffer[i][imag]))};
    const double power_density{std::pow(magnitude, 2) / (2 * A * freq)};
    const double log_val{-10 * std::log10(power_density)};
    (*in_out_data)[i] = log_val;
  }

  // Cleanup FFTW object.
  fftw_destroy_plan(p);
  fftw_cleanup();
}

namespace spectrogram {

template <std::size_t X, std::size_t Y = X>
static inline std::array<std::array<double, Y>, X> CreateSpectrogram(const std::vector<double> &source_signal) {
  static_assert((Y * 2) < max_fft_size);
  const double slope_m{1.0 / (fft_spectogram_min - fft_spectogram_max)};

  std::array<std::array<double, Y>, X> return_spectogram = {{}};

  const std::size_t increment_size = source_signal.size() / X;

  std::size_t source_idx{0U};
  for (auto window = return_spectogram.begin(); window != return_spectogram.end(); ++window) {
    const auto copy_size = std::min(X, source_signal.size() - source_idx);

    std::memcpy(scratch_buffer.data(), &source_signal[source_idx], sizeof(double) * copy_size);
    std::fill(scratch_buffer.begin() + copy_size, scratch_buffer.begin() + X, 0.0);

    OneSidedFFT(reinterpret_cast<std::array<double, X> *>(&scratch_buffer), SAMPLE_RATE);

    // Normalize to values between 0.0 and 1.0
    std::transform(scratch_buffer.begin(), scratch_buffer.begin() + X, window->begin(),
                   [&](const auto &val) { return static_cast<double>(std::clamp<double>(val * slope_m, 0.0, 1.0)); });

    source_idx += increment_size;
  }

  return return_spectogram;
}

template <std::size_t X, std::size_t Q = X * 25>
static inline std::array<std::array<double, X>, X> CreateMelSpectrogram(const std::vector<double> &source_signal) {
  const double slope_m = 1.0 / (fft_spectogram_min - fft_spectogram_max);
  const double log_max = std::log2(static_cast<double>(SAMPLE_RATE+1));

  std::array<std::array<double, X>, X> mel_spectogram{{}};
  auto source_idx{0U};
  const auto increment_size = source_signal.size() / X;

  for (auto x{0U}; x < X; ++x) {
    const auto copy_size = std::min(Q, source_signal.size() - source_idx);

    std::memcpy(scratch_buffer.data(), &source_signal[source_idx], sizeof(double) * std::min(Q, source_signal.size() - source_idx));
    std::fill(scratch_buffer.begin() + copy_size, scratch_buffer.begin() + Q, 0.0);

    OneSidedFFT(reinterpret_cast<std::array<double, Q> *>(&scratch_buffer), SAMPLE_RATE);

    // Map Q size FFT to X sized FFT using log2 transform.
    std::array<std::vector<double>, X> log_bins{};
    for (auto y{0U}; y < Q; ++y) {
      const auto mapped_y = static_cast<std::size_t>(std::floor(X * std::log2(y + 1) / log_max)); // TODO double check math.
      if (mapped_y < X) {
        log_bins[mapped_y].push_back(scratch_buffer[y]);
      }
    }
    for (auto y{0U}; y < X; ++y) {
      const double sum_val = std::accumulate(log_bins[y].begin(), log_bins[y].end(), 0);
      mel_spectogram[x][y] = std::clamp(slope_m * sum_val / log_bins[y].size(), 0.0, 1.0);
    }

    source_idx += increment_size;
  }

  return mel_spectogram;
}

} // namespace spectrogram
} // namespace fft

#endif /* INCLUDE_FFT_H_ */
